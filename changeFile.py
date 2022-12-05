import os, shutil

root="C:/dataset/glint_umd"
default_folder="imgs0"
alternative_folders_number=9

alternative_folders=[]
for i in range(alternative_folders_number):
    alternative_folders.append("imgs"+str(i+1))
    d=os.path.join(root,alternative_folders[i])
    if not os.path.exists(d): os.mkdir(d)

# for class_root,_,image_files  in os.walk(os.path.join(root,default_folder)):
    # for file in image_files:
    #     alternative_folder_id=hash(file)%alternative_folders_number
    #     if alternative_folder_id==0:continue
    #     destination=os.path.join(root,alternative_folders[alternative_folder_id-1],os.path.basename(class_root))
    #     if not os.path.exists(destination): os.mkdir(destination)
    #     shutil.move(os.path.join(class_root,file),os.path.join(destination,file))
if __name__ == "__main__":
    dir_name_list= [name for name in os.listdir(os.path.join(root,default_folder)) if os.path.isdir(os.path.join(root,default_folder,name))]
    for dir_name in dir_name_list:
        for alternative_folder in alternative_folders:
            # print(os.path.isdir(os.path.join(root,alternative_folder,dir_name)))
            if os.path.isdir(os.path.join(root,alternative_folder,dir_name)):continue
            os.mkdir(os.path.join(root,alternative_folder,dir_name))
