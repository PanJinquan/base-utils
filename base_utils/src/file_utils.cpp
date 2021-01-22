//
// Created by dm on 2021/1/15.
//

#include "file_utils.h"


std::string load_file(string path) {
    std::ifstream file(path, std::ios::in);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size = file.tellg();
        char *content = new char[size];

        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

void write_datatxt(string path, string data, bool bCover) {
    //fstream fout(path, ios::app);
    fstream fout;
    if (bCover) {
        fout.open(path);//默认是：ios_base::in | ios_base::out
    } else {
        fout.open(path, ios::app);//所有写入附加在文件末尾
    }
    fout << data << endl;
    fout.flush();
    fout.close();
}

bool file_exists(string path) {
    fstream _file;
    _file.open(path, ios::in);
    if (!_file) {
        return false;
    } else {
        return true;
    }
}


#ifdef _LINUX
#include <memory.h>
#include <dirent.h>
vector<string> get_files_list(string dirpath) {
    vector<string> allPath;
    DIR *dir = opendir(dirpath.c_str());
    if (dir == NULL)
    {
        cout << "opendir error" << endl;
        return allPath;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_DIR) {//It's dir
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            string dirNew = dirpath + separator + entry->d_name;
            vector<string> tempPath = getFilesList(dirNew);
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());

        }
        else {
            //cout << "name = " << entry->d_name << ", len = " << entry->d_reclen << ", entry->d_type = " << (int)entry->d_type << endl;
            string name = entry->d_name;
            string imgdir = dirpath + separator + name;
            //sprintf("%s",imgdir.c_str());
            allPath.push_back(imgdir);
        }

    }
    closedir(dir);
    //system("pause");
    return allPath;
}
#endif


#ifdef _WIN32//__WINDOWS_
#include <io.h>
vector<string> get_files_list(string dir)
{
    vector<string> allPath;
    // 在目录后面加上"\\*.*"进行第一次搜索
    string dir2 = dir + separator+"*.*";

    intptr_t handle;
    _finddata_t findData;

    handle = _findfirst(dir2.c_str(), &findData);
    if (handle == -1) {// 检查是否成功
        cout << "can not found the file ... " << endl;
        return allPath;
    }
    while (_findnext(handle, &findData) == 0)
    {
        if (findData.attrib & _A_SUBDIR) 是否含有子目录
        {
            //若该子目录为"."或".."，则进行下一次循环，否则输出子目录名，并进入下一次搜索
            if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
                continue;
            // 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
            string dirNew = dir + separator + findData.name;
            vector<string> tempPath = getFilesList(dirNew);
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());
        }
        else //不是子目录，即是文件，则输出文件名和文件的大小
        {
            string filePath = dir + separator + findData.name;
            allPath.push_back(filePath);
        }
    }
    _findclose(handle);    // 关闭搜索句柄
    return allPath;
}
#endif