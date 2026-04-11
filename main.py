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
from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey, Float, Date, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/smartfridge")
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM    = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 h

# ✅ 安全：从 .env 文件读取，Key 不暴露在代码里
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
# Pydantic Schemas
# ─────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class RegisterResponse(BaseModel):
    user_id:  int
    username: str
    email:    str

class LoginRequest(BaseModel):
    email:    EmailStr
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

class FridgeItemCreate(BaseModel):
    user_id:     int
    ingredient:  str
    quantity:    float
    unit:        str = "pcs"
    expiry_date: date

class FridgeItemUpdate(BaseModel):
    quantity:    Optional[float] = None
    expiry_date: Optional[date]  = None

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

class AIChatRequest(BaseModel):
    message:         str
    user_id:         Optional[str] = "user_1"
    conversation_id: Optional[str] = ""

class AIChatResponse(BaseModel):
    answer:          str
    conversation_id: str

# ✅ 新增：AI + 购物清单联动返回
class AIChatWithShoppingResponse(BaseModel):
    answer:            str
    conversation_id:   str
    added_to_shopping: List[str]

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
        item_id     = item.item_id,
        user_id     = item.user_id,
        ingredient  = item.ingredient,
        quantity    = item.quantity,
        unit        = item.unit,
        expiry_date = item.expiry_date,
        days_left   = days_left,
    )

def call_dify(query: str, user_id: str, conversation_id: str = "") -> dict:
    """统一的 Dify 调用函数"""
    res = req_lib.post(
        DIFY_API_URL,
        headers={
            "Authorization": f"Bearer {DIFY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": conversation_id or "",
            "user": f"user_{user_id}",
        },
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(UserModel).filter(UserModel.email == body.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")
    new_user = UserModel(
        username      = body.username,
        email         = body.email,
        password_hash = hash_password(body.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return RegisterResponse(user_id=new_user.user_id, username=new_user.username, email=new_user.email)

@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    token = create_access_token({"sub": str(user.user_id), "email": user.email})
    return LoginResponse(access_token=token, user_id=user.user_id)

@app.get("/tags", response_model=TagsResponse)
def get_tags(db: Session = Depends(get_db)):
    all_tags = db.query(TagModel).all()
    result: dict[str, List[TagOut]] = {"goal": [], "diet": [], "region": []}
    for tag in all_tags:
        key = tag.tag_type.lower()
        if key in result:
            result[key].append(TagOut(tag_id=tag.tag_id, tag_name=tag.tag_name))
    return TagsResponse(**result)

@app.post("/user/tags", response_model=UserTagsResponse, status_code=status.HTTP_201_CREATED)
def add_user_tags(body: UserTagsRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    existing_ids = {r.tag_id for r in db.query(UserTagModel).filter(UserTagModel.user_id == body.user_id).all()}
    new_records = [UserTagModel(user_id=body.user_id, tag_id=tid) for tid in body.tag_ids if tid not in existing_ids]
    if new_records:
        db.add_all(new_records)
        db.commit()
    tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == body.user_id).all()
    return UserTagsResponse(user_id=body.user_id, tags=[TagOut(tag_id=t.tag_id, tag_name=t.tag_name) for t in tags])

@app.get("/user/tags/{user_id}", response_model=UserTagsResponse)
def get_user_tags(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == user_id).all()
    return UserTagsResponse(user_id=user_id, tags=[TagOut(tag_id=t.tag_id, tag_name=t.tag_name) for t in tags])

# ==========================================
# 冰箱相关接口
# ==========================================

@app.get("/fridge/{user_id}", response_model=List[FridgeItemOut])
def get_fridge(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    items = db.query(FridgeItemModel).filter(FridgeItemModel.user_id == user_id).all()
    return [item_to_out(i) for i in items]

@app.post("/fridge", response_model=FridgeItemOut, status_code=status.HTTP_201_CREATED)
def add_fridge_item(body: FridgeItemCreate, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    new_item = FridgeItemModel(
        user_id     = body.user_id,
        ingredient  = body.ingredient,
        quantity    = body.quantity,
        unit        = body.unit,
        expiry_date = body.expiry_date,
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return item_to_out(new_item)

@app.delete("/fridge/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_fridge_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(FridgeItemModel).filter(FridgeItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found.")
    db.delete(item)
    db.commit()

# ==========================================
# 购物清单接口
# ==========================================

@app.get("/shopping/{user_id}", response_model=List[ShoppingItemOut])
def get_shopping(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    items = db.query(ShoppingItemModel).filter(ShoppingItemModel.user_id == user_id).all()
    return [
        ShoppingItemOut(
            item_id=i.item_id, user_id=i.user_id, ingredient=i.ingredient,
            quantity=i.quantity, unit=i.unit, is_purchased=bool(i.is_purchased)
        ) for i in items
    ]

@app.post("/shopping", response_model=ShoppingItemOut, status_code=status.HTTP_201_CREATED)
def add_shopping_item(body: ShoppingItemCreate, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.user_id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    new_item = ShoppingItemModel(
        user_id=body.user_id, ingredient=body.ingredient,
        quantity=body.quantity, unit=body.unit, is_purchased=0,
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return ShoppingItemOut(
        item_id=new_item.item_id, user_id=new_item.user_id, ingredient=new_item.ingredient,
        quantity=new_item.quantity, unit=new_item.unit, is_purchased=False
    )

@app.put("/shopping/{item_id}", response_model=ShoppingItemOut)
def update_shopping_item(item_id: int, body: ShoppingItemUpdate, db: Session = Depends(get_db)):
    item = db.query(ShoppingItemModel).filter(ShoppingItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found.")
    item.is_purchased = 1 if body.is_purchased else 0
    db.commit()
    db.refresh(item)
    return ShoppingItemOut(
        item_id=item.item_id, user_id=item.user_id, ingredient=item.ingredient,
        quantity=item.quantity, unit=item.unit, is_purchased=bool(item.is_purchased)
    )

@app.delete("/shopping/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shopping_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ShoppingItemModel).filter(ShoppingItemModel.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found.")
    db.delete(item)
    db.commit()

# ==========================================
# AI 聊天接口
# ==========================================

@app.post("/ai/chat", response_model=AIChatResponse)
def ai_chat(body: AIChatRequest):
    try:
        data = call_dify(body.message, body.user_id or "1", body.conversation_id or "")
        return AIChatResponse(
            answer=data.get("answer", "收到！"),
            conversation_id=data.get("conversation_id", ""),
        )
    except req_lib.exceptions.HTTPError as e:
        print(f"[Dify HTTP错误] status={e.response.status_code} body={e.response.text}")
        raise HTTPException(status_code=502, detail=f"Dify 返回错误：{e.response.status_code}")
    except Exception as e:
        print(f"[Dify 异常] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"AI 服务不可用：{str(e)}")


# ==========================================
# ✅ 新增：AI 生成菜谱 + 自动添加缺少食材到购物清单
# ==========================================

@app.post("/ai/chat-with-shopping", response_model=AIChatWithShoppingResponse)
def ai_chat_with_shopping(body: AIChatRequest, db: Session = Depends(get_db)):
    try:
        # 第一步：调用 Dify 获取菜谱回答
        dify_data = call_dify(body.message, body.user_id or "1", body.conversation_id or "")
        answer = dify_data.get("answer", "")
        conversation_id = dify_data.get("conversation_id", "")

        # 第二步：再调用 Dify 从菜谱文字里提取所需食材（结构化 JSON）
        extract_query = f"""
请从以下菜谱推荐内容中，提取出所有需要用到的食材。
只返回 JSON 数组，不要任何其他文字或解释，格式如下：
[{{"ingredient": "西红柿", "quantity": 2, "unit": "个"}}, {{"ingredient": "鸡蛋", "quantity": 3, "unit": "个"}}]

菜谱内容：
{answer}
"""
        extract_data = call_dify(extract_query, f"{body.user_id}_ext", "")
        extract_text = extract_data.get("answer", "[]")

        # 第三步：解析 JSON，容错处理
        ingredients_needed = []
        try:
            json_match = re.search(r'\[.*?\]', extract_text, re.DOTALL)
            if json_match:
                ingredients_needed = json.loads(json_match.group())
        except Exception as parse_err:
            print(f"[食材解析失败] {parse_err}，原始：{extract_text}")

        # 第四步：查询用户冰箱已有食材
        fridge_items = db.query(FridgeItemModel).filter(
            FridgeItemModel.user_id == int(body.user_id)
        ).all()
        fridge_names = {item.ingredient.strip() for item in fridge_items}

        # 第五步：过滤冰箱已有的，把缺的加入购物清单（去重）
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
                db.add(ShoppingItemModel(
                    user_id=int(body.user_id),
                    ingredient=name,
                    quantity=float(item.get("quantity", 1)),
                    unit=item.get("unit", "份"),
                    is_purchased=0,
                ))
                added.append(name)

        db.commit()

        return AIChatWithShoppingResponse(
            answer=answer,
            conversation_id=conversation_id,
            added_to_shopping=added,
        )

    except req_lib.exceptions.HTTPError as e:
        print(f"[Dify HTTP错误] status={e.response.status_code} body={e.response.text}")
        raise HTTPException(status_code=502, detail=f"Dify 返回错误：{e.response.status_code}")
    except Exception as e:
        print(f"[AI+购物 异常] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"服务不可用：{str(e)}")