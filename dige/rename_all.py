import os

def rename_all(folder_path):
    flag = 1
    filelist = os.listdir(folder_path)
    # 遍历列表进行重命名
    for i in filelist:
        new_name = 'python95_' + '%04d' % flag+'.data'
        flag+=1
        os.rename(os.path.join(folder_path, i), os.path.join(folder_path, new_name))

path=r'C:\迅雷下载\783900893\大学学姐\v'


rename_all(path)