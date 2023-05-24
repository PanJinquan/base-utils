//
// Created by Pan on 2021/1/15.
//

#include "file_utils.h"
#include<cstdio>
#include "debug.h"


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


void write_contents(string path, vector<string> contents, bool bCover) {
    //fstream fout(path, ios::app);
    fstream fout;
    if (bCover) {
        fout.open(path, ios_base::out);//默认是：ios_base::in | ios_base::out
    } else {
        fout.open(path, ios::app);//所有写入附加在文件末尾
    }
    int num = contents.size();
    for (int i = 0; i < num; ++i) {
        fout << contents.at(i) << endl;
    }
    fout.flush();
    fout.close();
}


vector<string> read_contents(string path) {
    ifstream infile;
    infile.open(path.data());   //将文件流对象与文件连接起来
    vector<string> contents;
    if (infile.is_open()) {
        string line;
        while (getline(infile, line)) {
            contents.push_back(line);
        }
    } else {
        LOGD("Failed to open file:%s", path.c_str());
    }
    infile.close();  //关闭文件输入流
    return contents;
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


std::string load_file(string path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
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
        LOGD("Failed to open file:%s", path.c_str());
        return "";
    }
}


int load_file(const char *path, std::string &file_string) {
    size_t uiSize = 0;
    size_t uiFileSize = 0;
    char *pStr = NULL;
    std::fstream fFile(path, (std::fstream::in | std::fstream::binary));
    if (fFile.is_open()) {
        fFile.seekg(0, std::fstream::end);
        uiSize = uiFileSize = (size_t) fFile.tellg();  // 获得文件大小
        fFile.seekg(0, std::fstream::beg);
        pStr = new char[uiSize + 1];

        if (NULL == pStr) {
            fFile.close();
            return 0;
        }

        fFile.read(pStr, uiFileSize);                // 读取uiFileSize字节
        fFile.close();
        pStr[uiSize] = '\0';
        file_string = pStr;
        delete[] pStr;
        return 0;
    }
    LOGD("Failed to open file:%s", path);
    return -1;
}


string get_basename(string path) {
    int index = path.find_last_of(separator);
    string name{""};
    if (index > -1) {
        name = path.substr(index + 1, path.length());
    }
    return name;
}

string get_parent(string path) {
    int index = path.find_last_of(separator);
    string parent{""};
    if (index > -1) {
        parent = path.substr(0, index);
    }
    return parent;
}

string get_subname(string path) {
    string parent = get_parent(path);
    string subname = get_basename(parent);
    return subname;
}


string get_postfix(string path, bool tolower) {
    std::string postfix = path.substr(path.find_last_of('.') + 1);
    if (tolower) {
        transform(postfix.begin(), postfix.end(), postfix.begin(), ::tolower);
        //transform(postfix.begin(), postfix.end(), postfix.begin(), ::toupper);
    }
    return postfix;
}

string path_joint(string path1, string path2) {
    return path1 + separator + path2;
}

bool remove_file(string file) {
    if (remove(file.c_str()) == 0) {
        return true;
    } else {
        return false;
    }
}
//#ifdef _LINUX
#ifdef PLATFORM_LINUX

#include <memory.h>
#include <dirent.h>

vector<string> get_files_list(string dirpath) {
    vector<string> allPath;
    DIR *dir = opendir(dirpath.c_str());
    if (dir == NULL) {
        LOGD("opendir error");
        return allPath;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {//It's dir
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            string dirNew = dirpath + separator + entry->d_name;
            vector<string> tempPath = get_files_list(dirNew);
            allPath.insert(allPath.end(), tempPath.begin(), tempPath.end());

        } else {
            //cout << "name = " << entry->d_name << ", len = " << entry->d_reclen << ", entry->d_type = " << (int)entry->d_type << endl;
            string name = entry->d_name;
            string imgdir = dirpath + separator + name;
            //LOGD("%s",imgdir.c_str());
            allPath.push_back(imgdir);
        }

    }
    closedir(dir);
    //system("pause");
    return allPath;
}


//#ifdef _WIN32//__WINDOWS_
#elif PLATFORM_WINDOWS//__WINDOWS_
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
            vector<string> tempPath = get_files_list(dirNew);
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

//#ifdef _LINUX
#elif PLATFORM_ANDROID
#include <memory.h>
#include <dirent.h>
vector<string> get_files_list(string dirpath) {
    vector<string> allPath;
    return allPath;
}

#endif