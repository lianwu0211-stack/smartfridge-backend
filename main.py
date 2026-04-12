import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, date
from typing import List, Optional

import bcrypt
import requests as req_lib
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/smartfridge")
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM    = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DIFY_API_URL = "https://api.dify.ai/v1/chat-messages"
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")

# ─────────────────────────────────────────
# DB setup
# ─────────────────────────────────────────
engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

# ─────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────
class UserModel(Base):
    __tablename__ = "USER"
    user_id       = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), nullable=False)
    email         = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    region        = Column(String(50), nullable=True)

class TagModel(Base):
    __tablename__ = "TAG"
    tag_id   = Column(Integer, primary_key=True, index=True)
    tag_name = Column(String(100), nullable=False)
    tag_type = Column(String(50), nullable=False)

class UserTagModel(Base):
    __tablename__ = "USER_TAG"
    user_id = Column(Integer, ForeignKey("USER.user_id"), primary_key=True)
    tag_id  = Column(Integer, ForeignKey("TAG.tag_id"),  primary_key=True)

class FridgeItemModel(Base):
    __tablename__ = "FRIDGE_ITEMS"
    item_id     = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("USER.user_id"), nullable=False)
    ingredient  = Column(String(100), nullable=False)
    quantity    = Column(Float, nullable=False)
    unit        = Column(String(20), nullable=False, default="pcs")
    expiry_date = Column(Date, nullable=False)

class ShoppingItemModel(Base):
    __tablename__ = "SHOPPING_ITEMS"
    item_id      = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("USER.user_id"), nullable=False)
    ingredient   = Column(String(100), nullable=False)
    quantity     = Column(Float, nullable=False)
    unit         = Column(String(20), nullable=False, default="pcs")
    is_purchased = Column(Integer, nullable=False, default=0)

Base.metadata.create_all(bind=engine)

# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class RegisterResponse(BaseModel):
    user_id: int
    username: str
    email: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      int

class TagOut(BaseModel):
    tag_id:   int
    tag_name: str
    class Config:
        from_attributes = True

class TagsResponse(BaseModel):
    goal:   List[TagOut]
    diet:   List[TagOut]
    region: List[TagOut]

class UserTagsRequest(BaseModel):
    user_id: int
    tag_ids: List[int]

class UserTagsResponse(BaseModel):
    user_id: int
    tags:    List[TagOut]

# ✅ 新增：覆盖更新 Tag（先删后写，解决 Tag 无法更新问题）
class UserTagsUpdateRequest(BaseModel):
    user_id: int
    tag_ids: List[int]

class FridgeItemCreate(BaseModel):
    user_id:     int
    ingredient:  str
    quantity:    float
    unit:        str = "pcs"
    expiry_date: date

class FridgeItemOut(BaseModel):
    item_id:     int
    user_id:     int
    ingredient:  str
    quantity:    float
    unit:        str
    expiry_date: date
    days_left:   int
    class Config:
        from_attributes = True

# ✅ 新增：批量删除冰箱食材
class FridgeBatchDeleteRequest(BaseModel):
    item_ids: List[int]

class ShoppingItemCreate(BaseModel):
    user_id:    int
    ingredient: str
    quantity:   float
    unit:       str = "pcs"

class ShoppingItemUpdate(BaseModel):
    is_purchased: bool

class ShoppingItemOut(BaseModel):
    item_id:      int
    user_id:      int
    ingredient:   str
    quantity:     float
    unit:         str
    is_purchased: bool
    class Config:
        from_attributes = True

# ✅ 新增：购物清单批量标记已购买
class ShoppingBatchPurchaseRequest(BaseModel):
    user_id:  int
    item_ids: List[int]

class AIChatRequest(BaseModel):
    message:         str
    user_id:         Optional[str] = "1"
    conversation_id: Optional[str] = ""

class AIChatResponse(BaseModel):
    answer:          str
    conversation_id: str

class AIChatWithShoppingResponse(BaseModel):
    answer:            str
    conversation_id:   str
    added_to_shopping: List[str]

# ✅ 新增：AI 读取冰箱推荐菜谱的返回（包含菜谱列表和所需食材）
class RecipeOption(BaseModel):
    name:        str
    ingredients: List[str]
    missing:     List[str]  # 冰箱里缺少的食材

class AiFridgeRecommendResponse(BaseModel):
    answer:          str
    conversation_id: str
    recipes:         List[RecipeOption]

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire  = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def item_to_out(item: FridgeItemModel) -> FridgeItemOut:
    days_left = (item.expiry_date - date.today()).days
    return FridgeItemOut(
        item_id=item.item_id, user_id=item.user_id, ingredient=item.ingredient,
        quantity=item.quantity, unit=item.unit, expiry_date=item.expiry_date, days_left=days_left,
    )

def call_dify(query: str, user_id: str, conversation_id: str = "") -> dict:
    res = req_lib.post(
        DIFY_API_URL,
        headers={"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"},
        json={"inputs": {}, "query": query, "response_mode": "blocking",
              "conversation_id": conversation_id or "", "user": f"user_{user_id}"},
        timeout=30,
    )
    res.raise_for_status()
    return res.json()

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(title="SmartFridge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ==========================================
# 用户认证
# ==========================================

@app.post("/register", response_model=RegisterResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(UserModel).filter(UserModel.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")
    new_user = UserModel(username=body.username, email=body.email, password_hash=hash_password(body.password))
    db.add(new_user); db.commit(); db.refresh(new_user)
    return RegisterResponse(user_id=new_user.user_id, username=new_user.username, email=new_user.email)

@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token({"sub": str(user.user_id), "email": user.email})
    return LoginResponse(access_token=token, user_id=user.user_id)

# ==========================================
# Tag 接口
# ==========================================

@app.get("/tags", response_model=TagsResponse)
def get_tags(db: Session = Depends(get_db)):
    all_tags = db.query(TagModel).all()
    result: dict = {"goal": [], "diet": [], "region": []}
    for tag in all_tags:
        key = tag.tag_type.lower()
        if key in result:
            result[key].append(TagOut(tag_id=tag.tag_id, tag_name=tag.tag_name))
    return TagsResponse(**result)

@app.post("/user/tags", response_model=UserTagsResponse, status_code=201)
def add_user_tags(body: UserTagsRequest, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    existing_ids = {r.tag_id for r in db.query(UserTagModel).filter(UserTagModel.user_id == body.user_id).all()}
    new_records = [UserTagModel(user_id=body.user_id, tag_id=tid) for tid in body.tag_ids if tid not in existing_ids]
    if new_records:
        db.add_all(new_records); db.commit()
    tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == body.user_id).all()
    return UserTagsResponse(user_id=body.user_id, tags=[TagOut(tag_id=t.tag_id, tag_name=t.tag_name) for t in tags])

# ✅ 新增：PUT /user/tags — 覆盖更新（先清空再写入，解决 Tag 无法更新问题）
@app.put("/user/tags", response_model=UserTagsResponse)
def update_user_tags(body: UserTagsUpdateRequest, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    # 先删除该用户所有旧 Tag
    db.query(UserTagModel).filter(UserTagModel.user_id == body.user_id).delete()
    db.commit()
    # 再写入新 Tag
    db.add_all([UserTagModel(user_id=body.user_id, tag_id=tid) for tid in body.tag_ids])
    db.commit()
    tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == body.user_id).all()
    return UserTagsResponse(user_id=body.user_id, tags=[TagOut(tag_id=t.tag_id, tag_name=t.tag_name) for t in tags])

@app.get("/user/tags/{user_id}", response_model=UserTagsResponse)
def get_user_tags(user_id: int, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == user_id).all()
    return UserTagsResponse(user_id=user_id, tags=[TagOut(tag_id=t.tag_id, tag_name=t.tag_name) for t in tags])

# ==========================================
# 冰箱接口
# ==========================================

@app.get("/fridge/{user_id}", response_model=List[FridgeItemOut])
def get_fridge(user_id: int, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    return [item_to_out(i) for i in db.query(FridgeItemModel).filter(FridgeItemModel.user_id == user_id).all()]

@app.post("/fridge", response_model=FridgeItemOut, status_code=201)
def add_fridge_item(body: FridgeItemCreate, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    new_item = FridgeItemModel(user_id=body.user_id, ingredient=body.ingredient,
                               quantity=body.quantity, unit=body.unit, expiry_date=body.expiry_date)
    db.add(new_item); db.commit(); db.refresh(new_item)
    return item_to_out(new_item)

@app.delete("/fridge/{item_id}", status_code=204)
def delete_fridge_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(FridgeItemModel).filter(FridgeItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    db.delete(item); db.commit()

# ✅ 新增：批量删除冰箱食材（多选删除）
@app.post("/fridge/batch-delete", status_code=204)
def batch_delete_fridge(body: FridgeBatchDeleteRequest, db: Session = Depends(get_db)):
    db.query(FridgeItemModel).filter(FridgeItemModel.item_id.in_(body.item_ids)).delete(synchronize_session=False)
    db.commit()

# ==========================================
# 购物清单接口
# ==========================================

@app.get("/shopping/{user_id}", response_model=List[ShoppingItemOut])
def get_shopping(user_id: int, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    items = db.query(ShoppingItemModel).filter(ShoppingItemModel.user_id == user_id).all()
    return [ShoppingItemOut(item_id=i.item_id, user_id=i.user_id, ingredient=i.ingredient,
                            quantity=i.quantity, unit=i.unit, is_purchased=bool(i.is_purchased)) for i in items]

@app.post("/shopping", response_model=ShoppingItemOut, status_code=201)
def add_shopping_item(body: ShoppingItemCreate, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    new_item = ShoppingItemModel(user_id=body.user_id, ingredient=body.ingredient,
                                 quantity=body.quantity, unit=body.unit, is_purchased=0)
    db.add(new_item); db.commit(); db.refresh(new_item)
    return ShoppingItemOut(item_id=new_item.item_id, user_id=new_item.user_id, ingredient=new_item.ingredient,
                           quantity=new_item.quantity, unit=new_item.unit, is_purchased=False)

@app.put("/shopping/{item_id}", response_model=ShoppingItemOut)
def update_shopping_item(item_id: int, body: ShoppingItemUpdate, db: Session = Depends(get_db)):
    item = db.query(ShoppingItemModel).filter(ShoppingItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    item.is_purchased = 1 if body.is_purchased else 0
    db.commit(); db.refresh(item)
    return ShoppingItemOut(item_id=item.item_id, user_id=item.user_id, ingredient=item.ingredient,
                           quantity=item.quantity, unit=item.unit, is_purchased=bool(item.is_purchased))

@app.delete("/shopping/{item_id}", status_code=204)
def delete_shopping_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ShoppingItemModel).filter(ShoppingItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    db.delete(item); db.commit()

# ✅ 新增：批量标记全部已购买（一键全选）
@app.post("/shopping/batch-purchase", status_code=200)
def batch_purchase_shopping(body: ShoppingBatchPurchaseRequest, db: Session = Depends(get_db)):
    db.query(ShoppingItemModel).filter(
        ShoppingItemModel.user_id == body.user_id,
        ShoppingItemModel.item_id.in_(body.item_ids)
    ).update({"is_purchased": 1}, synchronize_session=False)
    db.commit()
    return {"success": True}

# ==========================================
# AI 聊天接口
# ==========================================

@app.post("/ai/chat", response_model=AIChatResponse)
def ai_chat(body: AIChatRequest):
    try:
        data = call_dify(body.message, body.user_id or "1", body.conversation_id or "")
        return AIChatResponse(answer=data.get("answer", "收到！"), conversation_id=data.get("conversation_id", ""))
    except req_lib.exceptions.HTTPError as e:
        print(f"[Dify HTTP错误] {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Dify 错误：{e.response.status_code}")
    except Exception as e:
        print(f"[Dify 异常] {e}")
        raise HTTPException(status_code=503, detail=f"AI 服务不可用：{str(e)}")

# ✅ 新增：AI 读取冰箱 + 关联用户 Tag + 推荐菜谱（返回结构化菜谱列表）
@app.post("/ai/fridge-recommend", response_model=AiFridgeRecommendResponse)
def ai_fridge_recommend(body: AIChatRequest, db: Session = Depends(get_db)):
    try:
        uid = int(body.user_id or "1")

        # 第一步：读取冰箱食材
        fridge_items = db.query(FridgeItemModel).filter(FridgeItemModel.user_id == uid).all()
        fridge_names = [i.ingredient for i in fridge_items]

        # 第二步：读取用户 Tag 偏好
        user_tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == uid).all()
        tag_names = [t.tag_name for t in user_tags]

        # 第三步：构建带上下文的 prompt
        fridge_str = "、".join(fridge_names) if fridge_names else "冰箱是空的"
        tags_str   = "、".join(tag_names)   if tag_names   else "无特殊偏好"
        prompt = (
            f"我的冰箱里有：{fridge_str}。\n"
            f"我的饮食偏好是：{tags_str}。\n"
            f"用户说：{body.message}\n\n"
            f"请根据冰箱食材和偏好推荐 2-3 道菜谱，每道菜说明名称和所需全部食材。"
        )

        # 第四步：调用 Dify（传入 conversation_id 实现多轮记忆）
        dify_data = call_dify(prompt, str(uid), body.conversation_id or "")
        answer = dify_data.get("answer", "")
        conversation_id = dify_data.get("conversation_id", "")

        # 第五步：提取结构化菜谱列表
        extract_prompt = f"""
从以下菜谱推荐中，提取每道菜的名称和所需食材。
只返回 JSON 数组，不要其他文字：
[{{"name":"番茄炒蛋","ingredients":["西红柿","鸡蛋","盐","油"]}}]

菜谱内容：
{answer}
"""
        extract_data = call_dify(extract_prompt, f"{uid}_ext", "")
        extract_text = extract_data.get("answer", "[]")

        recipes_raw = []
        try:
            m = re.search(r'\[.*\]', extract_text, re.DOTALL)
            if m:
                recipes_raw = json.loads(m.group())
        except Exception as e:
            print(f"[菜谱解析失败] {e}")

        # 第六步：对每道菜，标出冰箱里缺少的食材
        fridge_set = set(fridge_names)
        recipes = []
        for r in recipes_raw:
            ingredients = r.get("ingredients", [])
            missing = [i for i in ingredients if i not in fridge_set]
            recipes.append(RecipeOption(name=r.get("name", ""), ingredients=ingredients, missing=missing))

        return AiFridgeRecommendResponse(answer=answer, conversation_id=conversation_id, recipes=recipes)

    except req_lib.exceptions.HTTPError as e:
        print(f"[Dify HTTP错误] {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Dify 错误：{e.response.status_code}")
    except Exception as e:
        print(f"[AI冰箱推荐异常] {e}")
        raise HTTPException(status_code=503, detail=f"服务不可用：{str(e)}")

@app.post("/ai/chat-with-shopping", response_model=AIChatWithShoppingResponse)
def ai_chat_with_shopping(body: AIChatRequest, db: Session = Depends(get_db)):
    try:
        dify_data = call_dify(body.message, body.user_id or "1", body.conversation_id or "")
        answer = dify_data.get("answer", "")
        conversation_id = dify_data.get("conversation_id", "")

        extract_query = f"""
请从以下菜谱推荐内容中，提取出所有需要用到的食材。
只返回 JSON 数组，不要任何其他文字或解释：
[{{"ingredient":"西红柿","quantity":2,"unit":"个"}}]

菜谱内容：
{answer}
"""
        extract_data = call_dify(extract_query, f"{body.user_id}_ext", "")
        extract_text = extract_data.get("answer", "[]")

        ingredients_needed = []
        try:
            m = re.search(r'\[.*?\]', extract_text, re.DOTALL)
            if m:
                ingredients_needed = json.loads(m.group())
        except Exception as e:
            print(f"[食材解析失败] {e}")

        fridge_names = {i.ingredient.strip() for i in db.query(FridgeItemModel).filter(
            FridgeItemModel.user_id == int(body.user_id)).all()}

        added = []
        for item in ingredients_needed:
            name = item.get("ingredient", "").strip()
            if not name or name in fridge_names:
                continue
            existing = db.query(ShoppingItemModel).filter(
                ShoppingItemModel.user_id == int(body.user_id),
                ShoppingItemModel.ingredient == name,
                ShoppingItemModel.is_purchased == 0,
            ).first()
            if not existing:
                db.add(ShoppingItemModel(user_id=int(body.user_id), ingredient=name,
                                         quantity=float(item.get("quantity", 1)),
                                         unit=item.get("unit", "份"), is_purchased=0))
                added.append(name)
        db.commit()

        return AIChatWithShoppingResponse(answer=answer, conversation_id=conversation_id, added_to_shopping=added)

    except req_lib.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dify 错误：{e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用：{str(e)}")