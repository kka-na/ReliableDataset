import os
import shutil

# 이동할 파일들이 들어있는 폴더
source_folder = '/home/kana/Documents/Dataset/WAYMO/assurance'

# 이동할 파일들이 들어있는 폴더에서 파일 찾기
for file_name in os.listdir(source_folder):
    if file_name.startswith('before_by_b') and file_name.endswith('.txt'):
        # 파일 이름에서 "by_b" 이후의 문자열을 가져와서 파일 이름 생성
        new_file_name = file_name.split('by_b')[1]

        # 파일 이동
        shutil.move(os.path.join(source_folder, file_name), os.path.join(f'{source_folder}/before_by_b/', new_file_name))

    elif file_name.startswith('before_by_a') and file_name.endswith('.txt'):
        # 파일 이름에서 "by_a" 이후의 문자열을 가져와서 파일 이름 생성
        new_file_name = file_name.split('by_a')[1]

        # 파일 이동
        shutil.move(os.path.join(source_folder, file_name), os.path.join(f'{source_folder}/before_by_a/', new_file_name))
