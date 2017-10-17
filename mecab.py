# -*- coding: utf-8 -*-

import os
import MeCab
import pandas as pd

def extract_csv_column_data(filepath, filename):

    print("CSVファイルからのデータ抽出開始")
    
    df = pd.read_csv(filepath)

    for key, column in df[["既知化DB番号", "既知化DB名", "対応方法"]].iterrows():
        name = "{0}{1}対応方法.txt".format(filename, column["既知化DB番号"])
        with open(name, "a") as f:
            f.write(column["既知化DB名"])
            f.write("\n")
            f.write(column["対応方法"])

    print("データ保存終了")

def wakatigaki(filepath, filename):

    print("分かち書き開始")

    files = os.listdir(filepath)
    mecab = MeCab.Tagger(" -Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd -Owakati")

    for file_ in files:
        with open("{0}{1}".format(filepath, file_), "r") as f:
            data = f.readline()
            name, _ = os.path.splitext(file_)
            wakatifile = "{0}{1}_wakati.txt".format(filename, name)

            while data:
                wakati = mecab.parse(data)
                with open(wakatifile, "a") as f2:
                    f2.write(wakati)
                data = f.readline()

    print("データ保存終了")

if __name__ == "__main__":

    print("# start")
    
    fp = "~/AI/OPC/LDAPython/Data/既知化DB.csv"
    fn = "/home/tamashiro/AI/OPC/LDAPython/Data/対応方法/"
    extract_csv_column_data(fp, fn)

    fp2 = "/home/tamashiro/AI/OPC/LDAPython/Data/対応方法/"
    fn2 = "/home/tamashiro/AI/OPC/LDAPython/Data/対応方法_wakati/"
    wakatigaki(fp2, fn2)
