import os
import glob
import pickle
import pandas as pd
import openpyxl
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np

def ddi_drug(input_file): #查看相互作用的药物是否都在drug_list里
    parsed = open('../test/ddi_druglist.txt', 'w+')
    drug = []
    count = 0
    drug_list = []
    drug_set = set()
    with open('../data/DrugList.txt', 'r') as fp:
        for line in fp:
            drug_list.append(line.strip())
    with open(input_file, 'r') as fp:
        next(fp)
        total_lines = sum(1 for line in fp)  # 获取文件总行数
        fp.seek(0)  # 重置文件指针到开头
        next(fp)  # 跳过第一行
        for line in tqdm(fp, total=total_lines, desc="Processing"):  # 添加进度条
            sptlist = line.strip().split('\t')
            drug1 = sptlist[0].strip()
            drug2 = sptlist[1].strip()
            label = sptlist[2].strip()
            if drug1 not in drug_list:
                drug_set.add(drug1)
            if drug1 not in drug_list:
                drug_set.add(drug2)
            if drug1 not in drug:
                drug.append(drug1)
                parsed.write(drug1)
                count += 1
            if drug2 not in drug:
                drug.append(drug2)
                parsed.write(drug2)
                count +=1
    print("记录ddi相互作用类型的药物共用：%d种",count)
    print("不在drug_list中的药物：")
    print(drug_set)
    parsed.close()
    return
# raw_ddi_file = '../data/DrugBank_known_ddi.txt'
# ddi_drug(raw_ddi_file)


# 不能用cuda 因为数据有string类型 narray只支持int float bool等 且一个narray中数据类型要一致
def parse_drug_input(input_file):
    parsed = open('../test/my_ddi_input.txt', 'w+')
    parsed.write('prescription  drug A  smilesA drugB   smilesB label\n')
    merged = pd.read_csv('../data/Drug_info_combined.csv')
    count = 0
    with open(input_file, 'r') as fp:
        next(fp)
        total_lines = sum(1 for line in fp)  # 获取文件总行数
        fp.seek(0)  # 重置文件指针到开头
        next(fp)  # 跳过第一行
        for line in tqdm(fp, total=total_lines, desc="Processing"):  # 添加进度条
            sptlist = line.strip().split('\t')
            drug1 = sptlist[0].strip()
            drug2 = sptlist[1].strip()
            label = sptlist[2].strip()
            smile1 = merged.loc[merged['Drug name'] == drug1]['Smiles'].values[0]
            smile2 = merged.loc[merged['Drug name'] == drug2]['Smiles'].values[0]
            line2 = str(count) + '\t' + drug1 + '\t' + smile1 + '\t' + drug2 + '\t' + smile2 + '\t' + str(label) + '\n'
            # print(line2)
            parsed.write(line2)
            count += 1
    parsed.close()
    return
# raw_ddi_file = '../data/DrugBank_known_ddi.txt'
# parse_drug_input(raw_ddi_file)

def calculate_structure_similarity(drug_dir, input_file, output_file, drug_list):
    drugbank_drugs = glob.glob(drug_dir + '*')  # 2386
    all_input_drug_info = {}
    with open(input_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            prescription = sptlist[0].strip()
            drug1 = sptlist[1].strip()
            smiles1 = sptlist[2].strip()
            drug2 = sptlist[3].strip()
            smiles2 = sptlist[4].strip()
            if drug1 not in all_input_drug_info:
                all_input_drug_info[drug1] = smiles1
            if drug2 not in all_input_drug_info:
                all_input_drug_info[drug2] = smiles2

    drug_similarity_info = {}
    for input_drug_id in all_input_drug_info:
        try:
            each_smiles = all_input_drug_info[input_drug_id]
            drug2_mol = Chem.MolFromSmiles(each_smiles)  # 根据smiles获取指纹
            drug2_mol = AllChem.AddHs(drug2_mol)
        except:
            continue
        drug_similarity_info[input_drug_id] = {}
        for each_drug_id1 in drugbank_drugs:
            drugbank_id = os.path.basename(each_drug_id1).split('.')[0]
            # Chem.MolFromMolFile 函数不是根据药物ID直接获取指纹的方法。它是使用指定的分子文件创建一个 Mol 对象
            # 对同一种药物分子，从smile和从文件获得mol对象是一致的
            drug1_mol = Chem.MolFromMolFile(each_drug_id1)

            drug1_mol = AllChem.AddHs(drug1_mol)

            fps = AllChem.GetMorganFingerprint(drug1_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(drug2_mol, 2)
            score = DataStructs.TanimotoSimilarity(fps, fps2)
            drug_similarity_info[input_drug_id][drugbank_id] = score

    df = pd.DataFrame.from_dict(drug_similarity_info)
    df = df.T
    df = df[drug_list]
    df.to_csv(output_file)


def calculate_pca(similarity_profile_file, output_file, pca_model):
    with open(pca_model, 'rb') as fid:
        pca = pickle.load(fid)
        df = pd.read_csv(similarity_profile_file, index_col=0)

        X = df.values
        X = pca.transform(X)

        new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(50)], index=df.index)
        new_df.to_csv(output_file)

# 得到模型输入文件：1_PC_1  1_PC_2  1_PC_3  2_PC_1  2_PC_2  2_PC_3
def generate_input_profile(input_file, pca_profile_file):
    df = pd.read_csv(pca_profile_file, index_col=0)
    all_drugs = []
    interaction_list = []
    with open(input_file, 'r') as fp:
        next(fp)  # 跳过第一行
        for line in fp:
            sptlist = line.strip().split('\t')
            prescription = sptlist[0].strip()
            drug1 = sptlist[1].strip()
            drug2 = sptlist[3].strip()
            all_drugs.append(drug1)
            all_drugs.append(drug2)
            if drug1 in df.index and drug2 in df.index:
                interaction_list.append([prescription, drug1, drug2])
                interaction_list.append([prescription, drug2, drug1])

    drug_feature_info = {}
    columns = ['PC_%d' % (i + 1) for i in range(50)]
    for row in df.itertuples():
        drug = row.Index
        feature = []
        drug_feature_info[drug] = {}
        for col in columns:
            val = getattr(row, col)  # 遍历columns列表中的每个列名col，使用getattr(row, col)获取该药物在当前特征列上的值，并将其存储到feature列表中
            feature.append(val)
            drug_feature_info[drug][col] = val

    new_col1 = ['1_%s' % (i) for i in columns]
    new_col2 = ['2_%s' % (i) for i in columns]

    DDI_input = {}
    for each_drug_pair in tqdm(interaction_list):
        prescription = each_drug_pair[0]
        drug1 = each_drug_pair[1]
        drug2 = each_drug_pair[2]
        key = '%s_%s_%s' % (prescription, drug1, drug2)

        DDI_input[key] = {}
        for col in columns:
            new_col = '1_%s' % (col)
            DDI_input[key][new_col] = drug_feature_info[drug1][col]

        for col in columns:
            new_col = '2_%s' % (col)
            DDI_input[key][new_col] = drug_feature_info[drug2][col]


    new_columns = []
    for i in [1, 2]:
        for j in range(1, 51):
            new_key = '%s_PC_%s' % (i, j)
            new_columns.append(new_key)

    df = pd.DataFrame.from_dict(DDI_input)
    df = df.T
    df = df[new_columns] # 选择特定的列
    print("数据集长度：")
    print(df.shape)
    # df.to_csv(output_file)
    return df

def generate_label(input_file,pca_profile_file):
    df = pd.read_csv(pca_profile_file, index_col=0)
    Label = []
    with open(input_file, 'r') as fp:
        next(fp)
        total_lines = sum(1 for line in fp)  # 获取文件总行数
        fp.seek(0)  # 重置文件指针到开头
        next(fp)  # 跳过第一行
        for line in tqdm(fp, total=total_lines, desc="Processing"):  # 添加进度条
            sptlist = line.strip().split('\t')
            drug1 = sptlist[1].strip()
            drug2 = sptlist[3].strip()
            label1 = label2= sptlist[5].strip()
            # print(label1)
            if drug1 in df.index and drug2 in df.index:
                Label.append(label1)
                Label.append(label2)
        # 创建包含 Label 数据的 DataFrame
    label_df = pd.DataFrame(Label, columns=['label'])
    print("标签长度：")
    # 将 DataFrame 保存为 CSV 文件
    label_df.to_csv('../test/my_label.csv', index=False)
    return

# concatDRKG的表征
def concat_drkg(input_file):
    with open('../data/drkg/index.pkl','rb') as file:
        index = pickle.load(file)
    print(len(index.keys()))
    node_emb = np.load('../data/drkg/DRKG_TransE_l2_entity.npy')
    print(node_emb.shape)
    non = set()
    all_drugs = []
    interaction_list = []
    with open(input_file, 'r') as fp:
        next(fp)  # 跳过第一行
        for line in fp:
            sptlist = line.strip().split('\t')
            prescription = sptlist[0].strip()
            drug1 = sptlist[1].strip()
            drug2 = sptlist[3].strip()
            all_drugs.append(drug1)
            all_drugs.append(drug2)
            if drug1 in index.keys() and drug2 in index.keys():
                interaction_list.append([prescription, drug1, drug2])
                interaction_list.append([prescription, drug2, drug1])
            elif drug1 not in index.keys():
                non.add(drug1)
            elif drug2 not in index.keys():
                non.add(drug2)
    print(non)
    # with open('./non.pkl','wb') as file:
    #           pickle.dump(non,file)
    drug_feature_info = {}
    columns = ['PC_%d' % (i + 1) for i in range(50)]
    DDI_input = {}
    for each_drug_pair in tqdm(interaction_list):
        prescription = each_drug_pair[0]
        drug1 = each_drug_pair[1]
        drug2 = each_drug_pair[2]
        key = '%s_%s_%s' % (prescription, drug1, drug2)

        DDI_input[key] = {}
        count = 0
        for col in columns:
            new_col = '1_%s' % (col)
            DDI_input[key][new_col] = node_emb[int(index[drug1])][count]
            count+=1

        count = 0
        for col in columns:
            new_col = '2_%s' % (col)
            DDI_input[key][new_col] = node_emb[int(index[drug2])][count]
            count+=1
    new_columns = []
    for i in [1, 2]:
        for j in range(1, 51):
            new_key = '%s_PC_%s' % (i, j)
            new_columns.append(new_key)

    df = pd.DataFrame.from_dict(DDI_input)
    df = df.T
    df = df[new_columns] # 选择特定的列
    print("数据集长度：")
    print(df.shape) # 443636 原：444254
    df.to_csv('./test/drkg.csv',index=False)
    return df
