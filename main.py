import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict

import bcrypt
import requests as req_lib
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, Date, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/smartfridge")
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM    = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
DIFY_API_URL = "https://api.dify.ai/v1/chat-messages"
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

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

class MealRecordModel(Base):
    __tablename__ = "MEAL_RECORDS"
    record_id    = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("USER.user_id"), nullable=False)
    meal_date    = Column(Date, nullable=False)
    recipe_name  = Column(String(200), nullable=False)
    ingredients  = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

# ─── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class RegisterResponse(BaseModel):
    user_id: int; username: str; email: str

class LoginRequest(BaseModel):
    email: EmailStr; password: str

class LoginResponse(BaseModel):
    access_token: str; token_type: str = "bearer"; user_id: int

class TagOut(BaseModel):
    tag_id: int; tag_name: str
    class Config: from_attributes = True

class TagsResponse(BaseModel):
    goal: List[TagOut]; diet: List[TagOut]; region: List[TagOut]

class UserTagsRequest(BaseModel):
    user_id: int; tag_ids: List[int]

class UserTagsResponse(BaseModel):
    user_id: int; tags: List[TagOut]

class UserTagsUpdateRequest(BaseModel):
    user_id: int; tag_ids: List[int]

class FridgeItemCreate(BaseModel):
    user_id: int; ingredient: str; quantity: float; unit: str = "pcs"; expiry_date: date

class FridgeItemUpdate(BaseModel):
    quantity: Optional[float] = None; expiry_date: Optional[date] = None

class FridgeItemOut(BaseModel):
    item_id: int; user_id: int; ingredient: str; quantity: float
    unit: str; expiry_date: date; days_left: int
    class Config: from_attributes = True

class FridgeBatchDeleteRequest(BaseModel):
    item_ids: List[int]

class ShoppingItemCreate(BaseModel):
    user_id: int; ingredient: str; quantity: float; unit: str = "pcs"

class ShoppingItemUpdate(BaseModel):
    is_purchased: bool

class ShoppingItemOut(BaseModel):
    item_id: int; user_id: int; ingredient: str
    quantity: float; unit: str; is_purchased: bool
    class Config: from_attributes = True

class ShoppingBatchPurchaseRequest(BaseModel):
    user_id: int; item_ids: List[int]

class AIChatRequest(BaseModel):
    message: str; user_id: Optional[str] = "1"; conversation_id: Optional[str] = ""

class AIChatResponse(BaseModel):
    answer: str; conversation_id: str

class AIChatWithShoppingResponse(BaseModel):
    answer: str; conversation_id: str; added_to_shopping: List[str]

class RecipeOption(BaseModel):
    name: str; ingredients: List[str]; missing: List[str]

class AiFridgeRecommendResponse(BaseModel):
    answer: str; conversation_id: str; recipes: List[RecipeOption]

# ✅ 新增：食材用量结构
class IngredientAmount(BaseModel):
    name: str; quantity: float; unit: str = "份"

# ✅ 更新：做法接口增加人数和结构化食材用量返回
class RecipeInstructionRequest(BaseModel):
    recipe_name:  str
    ingredients:  List[str]
    user_id:      Optional[str] = "1"
    conversation_id: Optional[str] = ""

class RecipeInstructionResponse(BaseModel):
    instructions:      str
    conversation_id:   str
    ingredient_amounts: List[IngredientAmount]  # 结构化食材用量

# ✅ 用餐记录
class MealRecordCreate(BaseModel):
    user_id:           int
    recipe_name:       str
    ingredients:       List[str]
    ingredient_amounts: List[IngredientAmount]  # 带数量的食材
    instructions:      str
    meal_date:         Optional[date] = None

class MealRecordOut(BaseModel):
    record_id: int; user_id: int; meal_date: date
    recipe_name: str; ingredients: List[str]; instructions: str
    class Config: from_attributes = True

# ✅ 新增：完成这餐时的冰箱扣减请求
class CompleteMealRequest(BaseModel):
    user_id:           int
    recipe_name:       str
    ingredients:       List[str]
    ingredient_amounts: List[IngredientAmount]
    instructions:      str

# ✅ 新增：冰箱扣减结果
class InsufficientItem(BaseModel):
    name:      str   # 食材名
    needed:    float # 需要数量
    available: float # 冰箱现有数量
    unit:      str

class CompleteMealResponse(BaseModel):
    status:       str  # "ok" | "insufficient"
    insufficient: List[InsufficientItem]  # 不足的食材列表
    record_id:    Optional[int] = None    # 成功时的记录id

# ✅ 新增：确认直接制作（食材不足但强制扣减）
class ForceCompleteMealRequest(BaseModel):
    user_id:           int
    recipe_name:       str
    ingredients:       List[str]
    ingredient_amounts: List[IngredientAmount]
    instructions:      str

# ─── Helpers ───────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

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

def ai_match_ingredient(recipe_ing: str, fridge_ing: str, user_id: str) -> bool:
    """用AI判断两个食材是否是同一种"""
    try:
        prompt = (
            f"请判断「{recipe_ing}」和「{fridge_ing}」是否是同一种食材？"
            f"只回答「是」或「否」，不要其他文字。"
        )
        data = call_dify(prompt, f"{user_id}_match", "")
        answer = data.get("answer", "").strip()
        return "是" in answer and "否" not in answer
    except:
        return False

def find_fridge_item(recipe_ing: str, fridge_items: list, user_id: str) -> Optional[FridgeItemModel]:
    """先精确匹配，再AI模糊匹配"""
    # 精确匹配
    for item in fridge_items:
        if item.ingredient == recipe_ing:
            return item
    # AI模糊匹配
    for item in fridge_items:
        if ai_match_ingredient(recipe_ing, item.ingredient, user_id):
            return item
    return None

# 单位标准化：将常见单位归一到可比较的组
UNIT_GROUPS = {
    "weight": {"克", "g", "千克", "kg", "公斤"},
    "volume": {"毫升", "ml", "升", "l", "L"},
    "count":  {"个", "只", "条", "片", "块", "颗", "根", "枚", "粒", "份", "pcs"},
}

def same_unit_group(u1: str, u2: str) -> bool:
    """判断两个单位是否属于同一类型（可以直接比数量）"""
    u1, u2 = u1.strip().lower(), u2.strip().lower()
    if u1 == u2:
        return True
    for group in UNIT_GROUPS.values():
        if u1 in group and u2 in group:
            return True
    return False

def normalize_quantity(qty: float, unit: str) -> float:
    """统一换算到基本单位（克/毫升/个），方便比较"""
    unit = unit.strip().lower()
    if unit in ("千克", "kg", "公斤"):
        return qty * 1000
    if unit in ("升", "l"):
        return qty * 1000
    return qty  # 克/毫升/个/份/pcs 直接返回

def deduct_fridge_items(user_id: int, ingredient_amounts: List[IngredientAmount],
                        db: Session, force: bool = False) -> List[InsufficientItem]:
    """
    扣减冰箱食材。
    force=False: 只检查，返回不足列表，不实际扣减
    force=True:  强制扣减，数量不足就扣到0删除
    单位不同类型时：只检查食材是否存在，不比较数量（无法换算）
    返回不足的食材列表
    """
    fridge_items = db.query(FridgeItemModel).filter(FridgeItemModel.user_id == user_id).all()
    insufficient = []

    for ia in ingredient_amounts:
        fridge_item = find_fridge_item(ia.name, fridge_items, str(user_id))

        if fridge_item is None:
            # 冰箱里完全没有这个食材
            insufficient.append(InsufficientItem(
                name=ia.name, needed=ia.quantity, available=0, unit=ia.unit
            ))
            continue

        # 判断单位是否可比较
        if not same_unit_group(ia.unit, fridge_item.unit):
            # 单位不同类（如"克" vs "pcs"），无法换算
            # 只要冰箱有这个食材就算有，不比数量，直接扣减（force时）
            print(f"[单位不兼容] {ia.name}: 需要 {ia.quantity}{ia.unit}，冰箱有 {fridge_item.quantity}{fridge_item.unit}，跳过数量检查")
            if force:
                fridge_item.quantity -= 1  # 象征性扣 1 单位
                if fridge_item.quantity <= 0:
                    db.delete(fridge_item)
            continue

        # 单位可比较，换算到基本单位再比较
        needed_base    = normalize_quantity(ia.quantity, ia.unit)
        available_base = normalize_quantity(fridge_item.quantity, fridge_item.unit)

        if available_base < needed_base:
            insufficient.append(InsufficientItem(
                name=ia.name, needed=ia.quantity,
                available=fridge_item.quantity, unit=ia.unit
            ))
            if force:
                db.delete(fridge_item)
        else:
            if force:
                # 扣减（换算回原单位）
                fridge_item.quantity -= ia.quantity
                if fridge_item.quantity <= 0:
                    db.delete(fridge_item)

    if force:
        db.commit()

    return insufficient

# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SmartFridge API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 用户认证 ──────────────────────────────────────────────────────────────────

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

# ── Tag ───────────────────────────────────────────────────────────────────────

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

@app.put("/user/tags", response_model=UserTagsResponse)
def update_user_tags(body: UserTagsUpdateRequest, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    db.query(UserTagModel).filter(UserTagModel.user_id == body.user_id).delete()
    db.commit()
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

# ── 冰箱 ──────────────────────────────────────────────────────────────────────

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
    if not item: raise HTTPException(status_code=404, detail="Item not found.")
    db.delete(item); db.commit()

@app.post("/fridge/batch-delete", status_code=204)
def batch_delete_fridge(body: FridgeBatchDeleteRequest, db: Session = Depends(get_db)):
    db.query(FridgeItemModel).filter(FridgeItemModel.item_id.in_(body.item_ids)).delete(synchronize_session=False)
    db.commit()

# ✅ 新增：更新冰箱食材数量和保质期（购物清单购买时用）
@app.put("/fridge/{item_id}", response_model=FridgeItemOut)
def update_fridge_item(item_id: int, body: FridgeItemUpdate, db: Session = Depends(get_db)):
    item = db.query(FridgeItemModel).filter(FridgeItemModel.item_id == item_id).first()
    if not item: raise HTTPException(status_code=404, detail="Item not found.")
    if body.quantity is not None: item.quantity = body.quantity
    if body.expiry_date is not None: item.expiry_date = body.expiry_date
    db.commit(); db.refresh(item)
    return item_to_out(item)

# ── 购物清单 ──────────────────────────────────────────────────────────────────

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
    if not item: raise HTTPException(status_code=404, detail="Item not found.")
    item.is_purchased = 1 if body.is_purchased else 0
    db.commit(); db.refresh(item)
    return ShoppingItemOut(item_id=item.item_id, user_id=item.user_id, ingredient=item.ingredient,
                           quantity=item.quantity, unit=item.unit, is_purchased=bool(item.is_purchased))

@app.delete("/shopping/{item_id}", status_code=204)
def delete_shopping_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ShoppingItemModel).filter(ShoppingItemModel.item_id == item_id).first()
    if not item: raise HTTPException(status_code=404, detail="Item not found.")
    db.delete(item); db.commit()

@app.post("/shopping/batch-purchase", status_code=200)
def batch_purchase_shopping(body: ShoppingBatchPurchaseRequest, db: Session = Depends(get_db)):
    db.query(ShoppingItemModel).filter(
        ShoppingItemModel.user_id == body.user_id,
        ShoppingItemModel.item_id.in_(body.item_ids)
    ).update({"is_purchased": 1}, synchronize_session=False)
    db.commit()
    return {"success": True}

# ── AI ────────────────────────────────────────────────────────────────────────

# ✅ 修复：/ai/chat 现在会读取用户 tags 和冰箱食材，注入到 prompt 中
@app.post("/ai/chat", response_model=AIChatResponse)
def ai_chat(body: AIChatRequest, db: Session = Depends(get_db)):
    try:
        uid = int(body.user_id or "1")

        # 读取用户 tags
        user_tags = (
            db.query(TagModel)
            .join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id)
            .filter(UserTagModel.user_id == uid)
            .all()
        )
        tag_names = [t.tag_name for t in user_tags]
        tags_str = "、".join(tag_names) if tag_names else "无特殊偏好"

        # 读取冰箱食材
        fridge_items = db.query(FridgeItemModel).filter(FridgeItemModel.user_id == uid).all()
        fridge_names = [i.ingredient for i in fridge_items]
        fridge_str = "、".join(fridge_names) if fridge_names else "冰箱是空的"

        # 拼接上下文 + 用户消息
        enriched_message = (
            f"[用户背景信息]\n"
            f"饮食偏好标签：{tags_str}\n"
            f"冰箱现有食材：{fridge_str}\n\n"
            f"[用户消息]\n{body.message}"
        )

        data = call_dify(enriched_message, str(uid), body.conversation_id or "")
        return AIChatResponse(
            answer=data.get("answer", "收到！"),
            conversation_id=data.get("conversation_id", "")
        )
    except req_lib.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dify 错误：{e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI 服务不可用：{str(e)}")

@app.post("/ai/fridge-recommend", response_model=AiFridgeRecommendResponse)
def ai_fridge_recommend(body: AIChatRequest, db: Session = Depends(get_db)):
    try:
        uid = int(body.user_id or "1")
        fridge_items = db.query(FridgeItemModel).filter(FridgeItemModel.user_id == uid).all()
        fridge_names = [i.ingredient for i in fridge_items]
        user_tags = db.query(TagModel).join(UserTagModel, TagModel.tag_id == UserTagModel.tag_id).filter(UserTagModel.user_id == uid).all()
        tag_names = [t.tag_name for t in user_tags]
        fridge_str = "、".join(fridge_names) if fridge_names else "冰箱是空的"
        tags_str   = "、".join(tag_names)   if tag_names   else "无特殊偏好"
        prompt = (
            f"我的冰箱里有：{fridge_str}。\n"
            f"我的饮食偏好是：{tags_str}。\n"
            f"用户说：{body.message}\n\n"
            f"请根据冰箱食材和偏好推荐 2-3 道菜谱，每道菜说明名称和所需全部食材。"
        )
        dify_data = call_dify(prompt, str(uid), body.conversation_id or "")
        answer = dify_data.get("answer", "")
        conversation_id = dify_data.get("conversation_id", "")
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
            if m: recipes_raw = json.loads(m.group())
        except Exception as e:
            print(f"[菜谱解析失败] {e}")
        fridge_set = set(fridge_names)
        recipes = []
        for r in recipes_raw:
            ingredients = r.get("ingredients", [])
            missing = [i for i in ingredients if i not in fridge_set]
            recipes.append(RecipeOption(name=r.get("name", ""), ingredients=ingredients, missing=missing))
        return AiFridgeRecommendResponse(answer=answer, conversation_id=conversation_id, recipes=recipes)
    except req_lib.exceptions.HTTPError as e:
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
            if m: ingredients_needed = json.loads(m.group())
        except Exception as e:
            print(f"[食材解析失败] {e}")
        fridge_names = {i.ingredient.strip() for i in db.query(FridgeItemModel).filter(
            FridgeItemModel.user_id == int(body.user_id)).all()}
        added = []
        for item in ingredients_needed:
            name = item.get("ingredient", "").strip()
            if not name or name in fridge_names: continue
            existing = db.query(ShoppingItemModel).filter(
                ShoppingItemModel.user_id == int(body.user_id),
                ShoppingItemModel.ingredient == name,
                ShoppingItemModel.is_purchased == 0,
            ).first()
            if not existing:
                db.add(ShoppingItemModel(user_id=int(body.user_id), ingredient=name,
                                         quantity=float(item.get("quantity") or 1),
                                         unit=item.get("unit", "份"), is_purchased=0))
                added.append(name)
        db.commit()
        return AIChatWithShoppingResponse(answer=answer, conversation_id=conversation_id, added_to_shopping=added)
    except req_lib.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dify 错误：{e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用：{str(e)}")

# ✅ 更新：获取做法 + 同时解析结构化食材用量（含人数）
@app.post("/ai/recipe-instructions", response_model=RecipeInstructionResponse)
def get_recipe_instructions(body: RecipeInstructionRequest):
    try:
        ingredients_str = "、".join(body.ingredients)
        extract_prompt = (
            f"请从「{body.recipe_name}」的食材列表中，给出每道菜的具体用量。\n"
            f"食材：{ingredients_str}\n\n"
            f"只返回JSON数组，不要其他文字：\n"
            f'[{{"name":"鸡蛋","quantity":2,"unit":"个"}},{{"name":"西红柿","quantity":3,"unit":"个"}}]'
        )
        data = call_dify(extract_prompt, body.user_id or "1", body.conversation_id or "")
        instructions = data.get("answer", "")
        conversation_id = data.get("conversation_id", "")

        # 第二步：解析结构化食材用量（根据人数）
        extract_prompt = (
            f"请从「{body.recipe_name}」的食材列表中，给出每道菜的具体用量。\n"
            f"食材：{ingredients_str}\n\n"
            f"只返回JSON数组，不要其他文字：\n"
            f'[{{"name":"鸡蛋","quantity":2,"unit":"个"}},{{"name":"西红柿","quantity":3,"unit":"个"}}]'
        )
        extract_data = call_dify(extract_prompt, f"{body.user_id}_amounts", "")
        extract_text = extract_data.get("answer", "[]")

        ingredient_amounts = []
        try:
            m = re.search(r'\[.*\]', extract_text, re.DOTALL)
            if m:
                raw_list = json.loads(m.group())
                for item in raw_list:
                    ingredient_amounts.append(IngredientAmount(
                        name=item.get("name", ""),
                        quantity=float(item.get("quantity") or 1),
                        unit=item.get("unit", "份"),
                    ))
        except Exception as e:
            print(f"[食材用量解析失败] {e}")
            # 降级：每种食材默认1份
            ingredient_amounts = [IngredientAmount(name=ing, quantity=1, unit="份") for ing in body.ingredients]

        return RecipeInstructionResponse(
            instructions=instructions,
            conversation_id=conversation_id,
            ingredient_amounts=ingredient_amounts,
        )
    except Exception as e:
        print(f"[做法获取异常] {e}")
        raise HTTPException(status_code=503, detail=f"服务不可用：{str(e)}")

# ✅ 新增：完成这餐（检查冰箱数量，返回不足列表）
@app.post("/meal/complete", response_model=CompleteMealResponse)
def complete_meal(body: CompleteMealRequest, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")

    # ✅ 调试日志：检查前端传来的食材用量
    print(f"[完成这餐] user_id={body.user_id}, recipe={body.recipe_name}")
    print(f"[完成这餐] ingredient_amounts({len(body.ingredient_amounts)}): {[f'{ia.name} {ia.quantity}{ia.unit}' for ia in body.ingredient_amounts]}")

    # ✅ 保护：ingredient_amounts 为空时，用 ingredients 列表降级构造默认用量
    if not body.ingredient_amounts and body.ingredients:
        print(f"[完成这餐] ⚠️ ingredient_amounts 为空，使用 ingredients 降级构造")
        body.ingredient_amounts = [IngredientAmount(name=ing, quantity=1, unit="份") for ing in body.ingredients]

    # 检查食材是否足够（不实际扣减）
    insufficient = deduct_fridge_items(body.user_id, body.ingredient_amounts, db, force=False)

    if insufficient:
        # 有食材不足，返回提示，不保存记录
        return CompleteMealResponse(status="insufficient", insufficient=insufficient)

    # 食材充足，正常扣减并保存记录
    deduct_fridge_items(body.user_id, body.ingredient_amounts, db, force=True)

    record = MealRecordModel(
        user_id      = body.user_id,
        meal_date    = date.today(),
        recipe_name  = body.recipe_name,
        ingredients  = json.dumps(body.ingredients, ensure_ascii=False),
        instructions = body.instructions,
    )
    db.add(record); db.commit(); db.refresh(record)
    return CompleteMealResponse(status="ok", insufficient=[], record_id=record.record_id)

# ✅ 新增：强制完成这餐（食材不足时直接扣减到0）
@app.post("/meal/force-complete", response_model=CompleteMealResponse)
def force_complete_meal(body: ForceCompleteMealRequest, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")

    # ✅ 保护：ingredient_amounts 为空时降级构造
    if not body.ingredient_amounts and body.ingredients:
        print(f"[强制完成] ⚠️ ingredient_amounts 为空，使用 ingredients 降级构造")
        body.ingredient_amounts = [IngredientAmount(name=ing, quantity=1, unit="份") for ing in body.ingredients]

    # 强制扣减（数量不足就扣到0删除）
    deduct_fridge_items(body.user_id, body.ingredient_amounts, db, force=True)

    record = MealRecordModel(
        user_id      = body.user_id,
        meal_date    = date.today(),
        recipe_name  = body.recipe_name,
        ingredients  = json.dumps(body.ingredients, ensure_ascii=False),
        instructions = body.instructions,
    )
    db.add(record); db.commit(); db.refresh(record)
    return CompleteMealResponse(status="ok", insufficient=[], record_id=record.record_id)

# ── 用餐记录查询 ──────────────────────────────────────────────────────────────

@app.post("/meal", response_model=MealRecordOut, status_code=201)
def create_meal_record(body: MealRecordCreate, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == body.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    record = MealRecordModel(
        user_id      = body.user_id,
        meal_date    = body.meal_date or date.today(),
        recipe_name  = body.recipe_name,
        ingredients  = json.dumps(body.ingredients, ensure_ascii=False),
        instructions = body.instructions,
    )
    db.add(record); db.commit(); db.refresh(record)
    return MealRecordOut(
        record_id=record.record_id, user_id=record.user_id, meal_date=record.meal_date,
        recipe_name=record.recipe_name,
        ingredients=json.loads(record.ingredients) if record.ingredients else [],
        instructions=record.instructions or "",
    )

@app.get("/meal/{user_id}", response_model=List[MealRecordOut])
def get_meal_records(user_id: int, db: Session = Depends(get_db)):
    if not db.query(UserModel).filter(UserModel.user_id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    records = db.query(MealRecordModel).filter(MealRecordModel.user_id == user_id).order_by(MealRecordModel.meal_date.desc()).all()
    return [MealRecordOut(
        record_id=r.record_id, user_id=r.user_id, meal_date=r.meal_date,
        recipe_name=r.recipe_name,
        ingredients=json.loads(r.ingredients) if r.ingredients else [],
        instructions=r.instructions or "",
    ) for r in records]

@app.get("/meal/{user_id}/{meal_date}", response_model=List[MealRecordOut])
def get_meal_by_date(user_id: int, meal_date: date, db: Session = Depends(get_db)):
    records = db.query(MealRecordModel).filter(
        MealRecordModel.user_id == user_id,
        MealRecordModel.meal_date == meal_date,
    ).all()
    return [MealRecordOut(
        record_id=r.record_id, user_id=r.user_id, meal_date=r.meal_date,
        recipe_name=r.recipe_name,
        ingredients=json.loads(r.ingredients) if r.ingredients else [],
        instructions=r.instructions or "",
    ) for r in records]

@app.delete("/meal/{record_id}", status_code=204)
def delete_meal_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(MealRecordModel).filter(MealRecordModel.record_id == record_id).first()
    if not record: raise HTTPException(status_code=404, detail="Record not found.")
    db.delete(record); db.commit()