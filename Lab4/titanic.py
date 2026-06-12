import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# PHẦN 1: DATA CLEANSING & FEATURE ENGINEERING
# ============================================================

# 1. Hàm load dữ liệu
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("D:\\PTDL&HS\\Lab4\\titanic_disaster.csv")
print("10 dòng đầu tiên:")
print(df.head(10))

# 2. Thống kê dữ liệu thiếu và trực quan hóa bằng heatmap
print("\nSố lượng giá trị thiếu trên từng cột:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap dữ liệu thiếu (trước xử lý)")
plt.show()

# Nhận xét:
# - Cột Age thiếu khoảng 20% dữ liệu -> cần xử lý (điền giá trị trung bình theo nhóm).
# - Cột Cabin thiếu rất nhiều (~77%) -> không thể điền giá trị hợp lý, sẽ xử lý
#   bằng cách trích xuất ký tự đầu (typeCabin) và gán "Unknown" cho dòng thiếu.
# - Cột Embarked thiếu rất ít (2 dòng) -> có thể bỏ qua hoặc điền giá trị phổ biến nhất.

# 3. Tách cột Name thành firstName và secondName, sau đó xóa cột Name
df[['secondName', 'firstName']] = df['Name'].str.split(',', n=1, expand=True)
df['secondName'] = df['secondName'].str.strip()
df['firstName'] = df['firstName'].str.strip()
df.drop(columns=['Name'], inplace=True)

# 4. Rút gọn dữ liệu cột Sex: male -> M, female -> F
df['Sex'] = df['Sex'].map({'male': 'M', 'female': 'F'})

# 5. Xử lý dữ liệu thiếu trên biến Age

# 5a. Sử dụng Boxplot để xem phân phối tuổi theo Pclass
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Phân phối tuổi theo hạng hành khách (Pclass)")
plt.show()

# Nhận xét:
# - Tuổi trung bình của hành khách hạng 1 cao nhất, hạng 3 thấp nhất.
# - Có sự khác biệt rõ rệt về tuổi trung bình giữa các nhóm Pclass.
# => Quyết định: thay thế giá trị Age bị thiếu bằng giá trị trung bình tuổi
#    THEO TỪNG NHÓM Pclass (thay vì dùng trung bình toàn bộ), vì điều này
#    phản ánh chính xác hơn đặc điểm của từng nhóm hành khách.

# 5b. Thay thế giá trị Age bị thiếu bằng trung bình tuổi theo Pclass
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

print("\nThông tin cột Age sau khi xử lý thiếu:")
print(df['Age'].describe())
print("\nSố giá trị thiếu còn lại của Age:", df['Age'].isnull().sum())

# Trực quan hóa lại dữ liệu Age sau khi xử lý bằng Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap dữ liệu thiếu (sau khi xử lý Age)")
plt.show()

# 6. Xây dựng biến AgeGroup
def age_group(age):
    if age <= 12:
        return "Kid"
    elif age <= 18:
        return "Teen"
    elif age <= 60:
        return "Adult"
    else:
        return "Older"

df['AgeGroup'] = df['Age'].apply(age_group)
df['AgeGroup'] = pd.Categorical(df['AgeGroup'],
                                 categories=["Kid", "Teen", "Adult", "Older"],
                                 ordered=True)

# 7. Tách namePrefix (Mr, Mrs, Miss, Master...) ra khỏi secondName
df['namePrefix'] = df['secondName'].str.extract(r'\s*([A-Za-z]+)\.')
df['secondName'] = df['secondName'].str.replace(r'\s*([A-Za-z]+)\.', '', regex=True).str.strip()

# 8. Tạo familySize = 1 + SibSp + Parch
df['familySize'] = 1 + df['SibSp'] + df['Parch']

# 9. Tạo đặc trưng Alone
df['Alone'] = np.where(df['familySize'] == 1, 1, 0)

# 10. Tách typeCabin (ký tự đầu của Cabin), thiếu -> "Unknown"
df['typeCabin'] = df['Cabin'].str[0]
df['typeCabin'] = df['typeCabin'].fillna("Unknown")

# 11. Loại bỏ dữ liệu thừa (trùng lặp) đối với hành khách xuất hiện ở cả
#     train.csv và test.csv -- giữ lại bản ghi trong train.
# (Giả định: nếu có 2 file train.csv và test.csv, ta merge và loại trùng theo PassengerId)
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop_duplicates(subset='PassengerId', keep='first')
    print("\nĐã loại bỏ dữ liệu trùng giữa train và test (nếu có).")
except FileNotFoundError:
    # Trong trường hợp chỉ có 1 file titanic_disaster.csv, loại trùng theo PassengerId
    df = df.drop_duplicates(subset='PassengerId', keep='first')

print("\nDữ liệu sau khi Data Cleansing & Feature Engineering:")
print(df.head())
print(df.info())

# ============================================================
# PHẦN 2: KHAI THÁC THÔNG TIN HỮU ÍCH - EDA
# ============================================================

# 12. Tỉ lệ sống sót và thiệt mạng theo từng nhóm giới tính
plt.figure(figsize=(6, 5))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Sống sót và thiệt mạng theo giới tính")
plt.xlabel("Giới tính")
plt.ylabel("Số lượng")
plt.legend(title="Survived", labels=["Thiệt mạng (0)", "Sống sót (1)"])
plt.show()

# Nhận xét: Tỉ lệ sống sót ở nữ giới (F) cao hơn rất nhiều so với nam giới (M).
# Điều này phù hợp với quy tắc "phụ nữ và trẻ em lên xuồng cứu sinh trước".

# 13. Hành khách sống sót theo từng nhóm Pclass
plt.figure(figsize=(6, 5))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Sống sót theo hạng hành khách (Pclass)")
plt.xlabel("Pclass")
plt.ylabel("Số lượng")
plt.legend(title="Survived", labels=["Thiệt mạng (0)", "Sống sót (1)"])
plt.show()

# Nhận xét: Hành khách hạng 1 có tỉ lệ sống sót cao nhất, hạng 3 có tỉ lệ
# sống sót thấp nhất -> địa vị kinh tế xã hội ảnh hưởng lớn đến khả năng sống sót.

# 14. Sống sót theo giới tính và nhóm tuổi
plt.figure(figsize=(8, 5))
sns.countplot(x='AgeGroup', hue='Survived', data=df, order=["Kid", "Teen", "Adult", "Older"])
plt.title("Sống sót theo nhóm tuổi")
plt.show()

g = sns.catplot(x='AgeGroup', hue='Survived', col='Sex', data=df,
                 kind='count', order=["Kid", "Teen", "Adult", "Older"])
g.fig.suptitle("Sống sót theo giới tính và nhóm tuổi")
plt.show()

# Nhận xét: Trẻ em (Kid) ở cả hai giới đều có tỉ lệ sống sót khá cao.
# Phụ nữ ở mọi nhóm tuổi có tỉ lệ sống sót cao hơn nam giới tương ứng.

# 15. Xác suất sống sót dựa trên thông tin nhóm đi cùng (familySize / Alone)
plt.figure(figsize=(8, 5))
sns.barplot(x='familySize', y='Survived', data=df, errorbar=None)
plt.title("Xác suất sống sót theo kích thước gia đình (familySize)")
plt.ylabel("Tỉ lệ sống sót")
plt.show()

plt.figure(figsize=(6, 5))
sns.barplot(x='Alone', y='Survived', data=df, errorbar=None)
plt.title("Xác suất sống sót: đi một mình (1) hay đi cùng nhóm (0)")
plt.xticks([0, 1], ["Đi cùng nhóm", "Đi một mình"])
plt.ylabel("Tỉ lệ sống sót")
plt.show()

# Nhận xét: Hành khách đi cùng gia đình nhỏ (2-4 người) có xác suất sống sót
# cao hơn hành khách đi một mình hoặc đi cùng nhóm quá đông.

# 16. Xác suất sống sót dựa trên thông tin giá vé (Fare)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=["Thấp", "Trung bình", "Khá", "Cao"])

plt.figure(figsize=(6, 5))
sns.barplot(x='FareBin', y='Survived', data=df, errorbar=None)
plt.title("Xác suất sống sót theo nhóm giá vé")
plt.ylabel("Tỉ lệ sống sót")
plt.show()

# Nhận xét: Hành khách mua vé giá cao có xác suất sống sót cao hơn rõ rệt
# so với hành khách mua vé giá thấp, một lần nữa cho thấy ảnh hưởng của
# địa vị kinh tế - xã hội tới khả năng sống sót.

# 17. Số lượng người thiệt mạng và sống sót theo Pclass và cảng (Embarked)
g = sns.catplot(x='Pclass', hue='Survived', col='Embarked', data=df, kind='count')
g.fig.suptitle("Sống sót/Thiệt mạng theo Pclass và cảng sẽ cập bến (Embarked)")
plt.show()

# Nhận xét: Cảng Cherbourg (C) có tỉ lệ sống sót cao hơn
# so với các cảng còn lại, một phần do tỉ lệ hành khách hạng 1 ở cảng này cao hơn.
# Cảng Southampton (S) có số lượng hành khách thiệt mạng nhiều nhất, đặc biệt
# ở nhóm hạng 3.

print("\nHoàn tất Lab 4!")