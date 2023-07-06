import data_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import mymodel
from tqdm import tqdm
import pickle


# 原始文件路径：Supplementary_Information/Supplementary Data 1.xlsx
# 输入数据格式：（数据条目数，100）维 为相互作用的两个药物相似度的各50个PCA降维维度 1_PC_1……1_PC_50 2_PC_1 ……2_PC_50

# 1 读取从第2行开始的数据， drugA drugB DDItype
# raw_ddi_file = '../data/DrugBank_known_ddi.txt'

# 2 处理成：
#   Prescription drugA smilesA drugB smilesB （DDI_input.txt)
# data_preprocessing.parse_drug_input(raw_ddi_file) #带label
# test/my_ddi_input.txt

# 3 calculate_structure_similarity(drug_dir, input_file, similarity_profile, drug_list)
# 用这个函数计算药物对中的药物A和B跟已知药物信息的药物的相似度 即与2386中已知药物的相似度文件
# test/similarity_profile

# 4 用pca降维
# 最终相似度只保留druglist里的2386种的降维为50个维度的文件
# test/PCA_transformed_similarity_profile.csv

# 5 generate_input_profile(input_file, pca_similarity_profile)
# 得到输入矩阵 注意格式是联合起来的# 0_current drug(vitamin c)_other drug a(vitamin a) 1_PC_1 1_PC_50 2_PC_1 2_PC_50
# 就是利用DDI_input文件和每种药物单独的相似度联合得到的 这个函数应该可以直接用 要注意记录每条记录的DDI_type 免得训练的时候弄错了
# 这里正反两个只要记录一个就好了吗？  最终->还是记录两个
# data_preprocessing.generate_label('../test/my_ddi_input.txt', '../data/drug_tanimoto_PCA50.csv')

# 6 划分训练集80%（其中的10%做验证） 测试集20% （可以最后一步做）
# 读取数据集和标签
# Read CSV files in smaller chunks
# Define the chunk size for reading the data
chunksize = 10000

# Read the dataset and labels in chunks
dataset_chunks = pd.read_csv('../test/concat_data.csv', chunksize=chunksize)
labels_chunks = pd.read_csv('../test/new_label.csv', chunksize=chunksize)

# Initialize empty DataFrames for train, validation, and test data
train_data = pd.DataFrame()
val_data = pd.DataFrame()
test_data = pd.DataFrame()

# Split the data into train, validation, and test sets in chunks
# for dataset_chunk, labels_chunk in zip(dataset_chunks, labels_chunks):
#     # Merge the dataset and labels chunk
#     data_chunk = pd.concat([dataset_chunk, labels_chunk], axis=1)
#
#     # Split the data chunk into train, validation, and test sets
#     train_val_chunk, test_chunk = train_test_split(data_chunk, test_size=0.2, random_state=42)
#     train_chunk, val_chunk = train_test_split(train_val_chunk, test_size=0.1, random_state=42)
#
#     # Append the chunk data to the corresponding DataFrames
#     train_data = train_data.append(train_chunk)
#     val_data = val_data.append(val_chunk)
#     test_data = test_data.append(test_chunk)
#
# # Save the split datasets as CSV files
# train_data.to_csv('../test/train_data2.csv', index=False)
# val_data.to_csv('../test/validation_data2.csv', index=False)
# test_data.to_csv('../test/test_data2.csv', index=False)
#
# print("Over")


# 7 输入模型进行训练
# 加载和预处理数据
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
train_data = pd.read_csv('../test/train_data2.csv')
val_data = pd.read_csv('../test/validation_data2.csv') # 第一行作为列名不需要剔除
test_data = pd.read_csv('../test/test_data2.csv')
train_features, train_labels = train_data.iloc[:, 1:-1], train_data.iloc[:, -1] # 每行的第二列到倒数第二列为特征 最后一列是标签
print("train_features:")
print(train_features.shape) #原：(319862, 100) 剔除non:(319417, 100)
print("validation.shape:")
print(val_data.shape) #原：(319862, 100) 剔除non:(319417, 100)
print("train_shape:")
print(test_data) #原：(319862, 100) 剔除non:(319417, 100)
val_features, val_labels = val_data.iloc[:, 1:-1], val_data.iloc[:, -1]
test_features, test_labels = test_data.iloc[:, 1:-1], test_data.iloc[:, -1]

# 转换数据为张量并移动到CUDA设备上
train_features = torch.tensor(train_features.values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels.values, dtype=torch.float32).to(device)
val_features = torch.tensor(val_features.values, dtype=torch.float32).to(device)
val_labels = torch.tensor(val_labels.values, dtype=torch.float32).to(device)
test_features = torch.tensor(test_features.values, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels.values, dtype=torch.float32).to(device)

# 张量对象没有str属性 要先转回为dataframe对象
# 将标签转换为Pandas Series对象 需要先把数据放到内存而不是cuda上转换
train_labels = train_labels.cpu().numpy()
train_labels = pd.Series(train_labels).astype(str)
val_labels = val_labels.cpu().numpy()
val_labels = pd.Series(val_labels).astype(str)
test_labels = test_labels.cpu().numpy()
test_labels = pd.Series(test_labels).astype(str)

# 使用MultiLabelBinarizer将标签转换为二进制型
with open('../data/multilabelbinarizer.pkl', 'rb') as fid:
    mlb = pickle.load(fid)
# mlb = MultiLabelBinarizer()
train_labels = torch.tensor(mlb.fit_transform(train_labels.str.split()), dtype=torch.float32).to(device)
val_labels = torch.tensor(mlb.transform(val_labels.str.split()), dtype=torch.float32).to(device)
test_labels = torch.tensor(mlb.transform(test_labels.str.split()), dtype=torch.float32).to(device)

# 初始化模型并移动到CUDA设备上
model = mymodel.MyModel().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练参数
num_epochs = 10
batch_size = 32
num_batches = len(train_data) // batch_size # 9995

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0

    # 每个epoch迭代训练数据的批次
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size

            # 提取批次数据并移动到CUDA设备上
            batch_features = train_features[batch_start:batch_end].to(device) #(32,100)
            batch_labels = train_labels[batch_start:batch_end].to(device) # (32,1)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_features)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失
            running_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

    # 计算平均损失
    epoch_loss = running_loss / num_batches

    # 在验证集上评估模型
    model.eval()
    val_outputs = model(val_features)
    val_predictions = torch.round(val_outputs)
    val_accuracy = accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().detach().numpy())

    # 在测试集上评估模型
    test_outputs = model(test_features)
    test_predictions = torch.round(test_outputs)
    test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().detach().numpy())

    # 打印当前epoch的损失和验证集准确率
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}' )

#torch.save(model.state_dict(), '../test/mymodel.pkl')
