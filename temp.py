import sys
import subprocess
import os
import time
import argparse
import random
from pathlib import Path

# ==========================================
# 0. 自動化環境配置
# ==========================================
def setup_environment():
    required_packages = {
        "pandas": "pandas",
        "numpy": "numpy<2.0",
        "openpyxl": "openpyxl",
        "mlxtend": "mlxtend==0.23.1",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "tabulate": "tabulate"
    }
    for lib_name, install_name in required_packages.items():
        try:
            __import__(lib_name)
        except ImportError:
            print(f"📦 正在安裝 {install_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

setup_environment()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tabulate import tabulate

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================
# 1. 資料讀取與前置處理
# ==========================================
def preprocess_data(file_path: str, min_items_per_txn: int = 1):
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 '{file_path}'")
        print(f"   請確認您的檔案路徑: {os.getcwd()}")
        return None

    print(f">>> [1] 正在讀取數據: {file_path}")
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        df = pd.read_csv(file_path)

    # 剔除退貨與註銷 (QUANTITY <= 0)
    original_count = len(df)
    df = df[df["QUANTITY"] > 0]
    df["ITEM_ID"] = df["ITEM_ID"].astype(str)
    
    filtered_count = len(df)
    print(f"    - 已剔除 {original_count - filtered_count} 筆退貨/異常資料")

    # 轉換為交易格式
    transactions = df.groupby("INVOICE_NO")["ITEM_ID"].apply(list).values.tolist()
    
    if min_items_per_txn > 1:
        transactions = [t for t in transactions if len(t) >= min_items_per_txn]

    print(f"    - 有效交易筆數: {len(transactions)}")
    return transactions

# ==========================================
# 2. 建立稀疏矩陣
# ==========================================
def build_sparse_onehot(transactions: list[list]) -> pd.DataFrame:
    print("\n>>> [2] 正在建立 Sparse One-Hot 交易矩陣...")
    te = TransactionEncoder()
    oht = te.fit(transactions).transform(transactions, sparse=True)
    df_onehot = pd.DataFrame.sparse.from_spmatrix(oht, columns=te.columns_)
    df_onehot.columns = [str(c) for c in df_onehot.columns]
    df_onehot = df_onehot.astype("Sparse[bool]")
    n_txn, n_items = df_onehot.shape
    print(f"    - 交易數: {n_txn:,}, 品項數: {n_items:,}")
    return df_onehot

# ==========================================
# 3. 雙演算法比較
# ==========================================
def run_mining_algorithms(df_onehot: pd.DataFrame, min_support: float):
    start = time.time()
    frequent_ap = apriori(df_onehot, min_support=min_support, use_colnames=True)
    time_ap = time.time() - start

    start = time.time()
    frequent_fp = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
    time_fp = time.time() - start

    return {
        "frequent_fp": frequent_fp,
        "time_ap": time_ap,
        "time_fp": time_fp,
        "n_itemsets": len(frequent_fp)
    }

# ==========================================
# 4. 冗餘規則剔除
# ==========================================
def filter_redundant_rules(rules: pd.DataFrame) -> pd.DataFrame:
    if rules is None or rules.empty:
        return rules

    rules = rules.copy()
    rules["antecedents_set"] = rules["antecedents"].apply(frozenset)
    keep_mask = np.ones(len(rules), dtype=bool)
    
    for i, row in rules.iterrows():
        current_ant = row["antecedents_set"]
        current_con = row["consequents"]
        current_conf = row["confidence"]
        same_consequent_rules = rules[rules["consequents"] == current_con]
        
        for _, other_row in same_consequent_rules.iterrows():
            if i == _: continue
            other_ant = other_row["antecedents_set"]
            other_conf = other_row["confidence"]
            
            if other_ant.issubset(current_ant) and other_ant != current_ant:
                if other_conf >= current_conf:
                    keep_mask[i] = False
                    break
                    
    filtered = rules.loc[keep_mask].drop(columns=["antecedents_set"]).reset_index(drop=True)
    return filtered

# ==========================================
# 5. 推薦功能
# ==========================================
def recommend_products(purchased_items: list, rules_df: pd.DataFrame, verbose: bool = True):
    if rules_df is None or rules_df.empty:
        if verbose: print("    ⚠️ 目前沒有規則，無法進行推薦。")
        return []

    purchased_set = set(map(str, purchased_items))
    recommendations = set()
    
    for _, rule in rules_df.iterrows():
        ant_raw = rule["antecedents"]
        con_raw = rule["consequents"]
        antecedents = set(ant_raw) if isinstance(ant_raw, (frozenset, set)) else set(ant_raw)
        consequents = set(con_raw) if isinstance(con_raw, (frozenset, set)) else set(con_raw)

        if antecedents.issubset(purchased_set):
            recommendations.update(consequents)

    recommendations -= purchased_set
    rec_list = sorted(list(recommendations))

    if verbose:
        if rec_list:
            print(f"推薦您購買: {', '.join(rec_list)}")
        else:
            print(" 無相符規則，暫無推薦產品。")
        
    return rec_list

# ==========================================
# 6. 實驗與圖表繪製
# ==========================================
def run_experiments(df_onehot, transactions):
    # 改為直接存檔，不使用 outputs 資料夾
    print("\n>>> [3] 正在進行參數影響分析實驗 (包含繪圖)...")
    
    supports = [0.001, 0.002, 0.005]
    confidences = [0.1, 0.3, 0.5]
    
    results = []
    test_baskets = [t for t in transactions if len(t) >= 2][:3]
    
    for s in supports:
        mining_res = run_mining_algorithms(df_onehot, min_support=s)
        frequent_itemsets = mining_res["frequent_fp"]
        
        for c in confidences:
            if frequent_itemsets.empty:
                rules_count = 0
                filtered_count = 0
                rec_count = 0
            else:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=c)
                rules_count = len(rules)
                filtered_rules = filter_redundant_rules(rules)
                filtered_count = len(filtered_rules)
                
                unique_recs = set()
                for basket in test_baskets:
                    recs = recommend_products(basket, filtered_rules, verbose=False)
                    unique_recs.update(recs)
                rec_count = len(unique_recs)
            
            results.append({
                "Support": s,
                "Confidence": c,
                "Time(Apriori)": mining_res["time_ap"],
                "Time(FP-Growth)": mining_res["time_fp"],
                "Speedup": mining_res["time_ap"] / mining_res["time_fp"] if mining_res["time_fp"] > 0 else 0,
                "Raw Rules": rules_count,
                "Filtered Rules": filtered_count,
                "Rec Products": rec_count
            })
            
    results_df = pd.DataFrame(results)
    print("\n📊 參數影響分析結果表:")
    print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))
    
    # 存檔 - 直接存 CSV
    results_df.to_csv("experiment_summary.csv", index=False, encoding="utf-8-sig")
    
    # 繪圖 - 直接存 png
    plt.figure(figsize=(10, 5))
    df_plot = results_df.drop_duplicates(subset=["Support"])
    plt.plot(df_plot["Support"], df_plot["Time(Apriori)"], marker='o', label='Apriori')
    plt.plot(df_plot["Support"], df_plot["Time(FP-Growth)"], marker='s', label='FP-Growth')
    plt.xlabel('Minimum Support')
    plt.ylabel('Time (seconds)')
    plt.title('Algorithm Runtime Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_runtime.png")
    
    plt.figure(figsize=(10, 5))
    df_plot_2 = results_df[results_df["Support"] == 0.001]
    plt.bar(df_plot_2["Confidence"].astype(str), df_plot_2["Filtered Rules"], color='skyblue')
    plt.xlabel('Minimum Confidence')
    plt.ylabel('Number of Rules')
    plt.title('Rules Count vs Confidence (Support=0.001)')
    plt.savefig("plot_rules.png")

    return results_df

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    file_name = "交易資料集(2).xlsx" # 檔名
    
    # 1. 前置處理
    transactions = preprocess_data(file_name)
    
    if transactions:
        # 2. 建立稀疏矩陣
        df_onehot = build_sparse_onehot(transactions)
        
        # 3. 執行完整實驗
        summary_df = run_experiments(df_onehot, transactions)
        
        # 4. 產生最終最佳模型
        print("\n>>> [4] 產生最終模型 (Support=0.001, Conf=0.3)...")
        res = run_mining_algorithms(df_onehot, min_support=0.001)
        freq_items = res["frequent_fp"]
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.3)
        final_rules = filter_redundant_rules(rules)
        
        # 存檔處理 (CSV + Excel)
        output_csv = "mining_results.csv"
        output_xlsx = "mining_results.xlsx"
        
        # 處理輸出格式
        save_rules = final_rules.copy()
        save_rules['antecedents'] = save_rules['antecedents'].apply(lambda x: ','.join(list(map(str, x))))
        save_rules['consequents'] = save_rules['consequents'].apply(lambda x: ','.join(list(map(str, x))))
        
        # 儲存 CSV
        save_rules.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"    - [CSV] 規則已存檔至: {os.path.abspath(output_csv)}")
        
        # 儲存 Excel (因為您提到想要 Excel 檔)
        try:
            save_rules.to_excel(output_xlsx, index=False)
            print(f"    - [Excel] 規則已存檔至: {os.path.abspath(output_xlsx)}")
        except Exception as e:
            print(f"    [Excel 存檔失敗 (可能缺少 openpyxl): {e}")

        # 5. 推薦測試 (保證命中版)
        print("\n>>> [5] 推薦系統測試：")
        if not final_rules.empty:
            first_rule_antecedents = list(final_rules.iloc[0]['antecedents'])
            test_basket = list(first_rule_antecedents)
            print(f"    模擬顧客購買了: {test_basket}")
            recommend_products(test_basket, final_rules)
        else:
            print("沒有產生任何規則，無法進行推薦測試。")
        
    print("\n✨ 全部完成！檔案已儲存於程式所在目錄。")