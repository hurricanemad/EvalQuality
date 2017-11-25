#include <iostream>
#include <fstream>
#include <string>
#include <getopt.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

static const char sccShort_Option[] = "s:d:l:c:a:";
static const struct option scslong_options[] = {
    {"source",      required_argument,   NULL, 's'},
    {"destination", required_argument,   NULL, 'd'},
    {"evaluation",  required_argument,   NULL, 'l'},
    {"classname",   required_argument,   NULL, 'c'},
    {"add",         required_argument,   NULL, 'a'},
};

static string ssstrSrcPath = "";
static string ssstrDstPath = "";
static string ssstrEvalList = "";
static string ssstrClsName = "";
static string ssstrAdd = "";

static void usage(FILE* fp,
                  int argc,
                  char** argv){
    fprintf(fp,
            "Usage: %s [options] \n\n"
            "Options:\n"
            "-s | --source               Source Label Path\n"
            "-d | --destination          Destination Label Path\n"
            "-l | --evaluation           Evaluation Labels' NameList Path\n"
            "-c | --classname            Class Name List Path\n"
            "-a | --add                  Add Operation Mark\n"
            "",
            argv[0]
    );
    
}

struct InputDataPair{
    string m_strDataName;
    string m_strLabelName;
    bool m_bHasLabels;
};

void ParseClassName(const string&, vector<string>&);
void IntersectionAndUnion(const Mat&, const Mat&, const vector<string> , vector<Mat>&, vector<Mat>&);
void PixelAccuracy(const Mat&, const Mat&, long& , long& );
void FigureIOU(const vector<Mat>&, const vector<Mat>&, vector<double>&, vector<double>&);
vector<int>vnIOUSum;

int main(int argc, char **argv) {
        for(;;){
        int nIndex;
        int c;
        
        c = getopt_long (argc, argv,
                         sccShort_Option, scslong_options, 
                         &nIndex);
        
        if(-1 == c)
            break;

       
        switch(c){
            case 0: /*getopt_long() flag*/
                break;
            case 's':
                ssstrSrcPath = optarg;
                break;
            case 'd':
                ssstrDstPath = optarg;
                break;
            case 'l':
                ssstrEvalList = optarg;
                break;
            case 'c':
                ssstrClsName = optarg;
                break;
            case 'a':
                ssstrAdd = optarg;
                break;
            default:
                usage(stderr, argc, argv);
                exit(EXIT_FAILURE);                
        }
    }
    //cout << ssstrEvalList << endl;
    
    if(ssstrEvalList == ""){
        cerr << "Evaluation labels' nameList is empty! Please set -l option!" <<endl;
        exit(1);
    }
    
    if(ssstrSrcPath == ""){
        cerr << "Source Label Path is empty! Please set -s option!" << endl;
        exit(1);
    }else{
        if(ssstrSrcPath.c_str()[ssstrSrcPath.length() - 1] == '/'){
            ssstrSrcPath = ssstrSrcPath.substr(0, ssstrSrcPath.length() - 1);
        }
    }
    
    if(ssstrDstPath == ""){
        cerr << "Destination Label Path is empty! Please set -d option!" << endl;
        exit(1);
    }else{
        if(ssstrDstPath.c_str()[ssstrDstPath.length() - 1] == '/'){
            ssstrDstPath = ssstrDstPath.substr(0, ssstrDstPath.length() - 1);
        }
    }
    
    if(ssstrClsName == ""){
        cerr << "Class Name List Path" << endl;
        exit(-1);
    }else{
        string strTempEvalList = ssstrEvalList;
        reverse(strTempEvalList.begin(), strTempEvalList.end());
        size_t szELPos = strTempEvalList.find("/");
        string EvalListPath = ssstrEvalList.substr(0, ssstrEvalList.length() - szELPos);
        ssstrClsName = EvalListPath + ssstrClsName;
        //cout << ssstrClsName << endl;
    }
    
    
    
    ifstream isEvalList(ssstrEvalList.c_str());
    string strTempLine;
    string strDataName;
    Mat matDataMat;
    Mat matLabelMat;
    long lLabelNum = 0l, lCorrectNum = 0l;
    double dAccuracy;
    double dTotalAccuracy = 0.0;
    vector<Mat> vmatIntersectionMat;
    vector<Mat> vmatUnionMat;
    vector<double>vdIOU;
    vector<double>vdInterScore;
    vector<double>vdUnionScore;
    vector<double>vdTotalIOU;
    double dTotalIOU = 0.0;
    string strLabelName("");
    size_t szDelimes;
    int nFile = 0;
    //bool bHasLabels;
    //vector<InputDataPair>vidDataPair;
    InputDataPair idTempDataPair;
    vector<string>vstrClsName;
    
    ParseClassName(ssstrClsName, vstrClsName);
    //cout << vstrClsName.size() <<endl;
    vdTotalIOU.resize(vstrClsName.size());
    vnIOUSum.resize(vstrClsName.size());
    for(int n=0; n < vdTotalIOU.size(); n++){
        vdTotalIOU[n] = 0.0;
        vnIOUSum[n] = 0;
    }
    
    if(isEvalList.is_open()){
        while(getline(isEvalList, strTempLine)){
            if(string::npos != strTempLine.find(" ")){
                szDelimes = strTempLine.find(" ");
                idTempDataPair.m_strDataName = strTempLine.substr(0, szDelimes);
                idTempDataPair.m_strLabelName = strTempLine.substr(szDelimes + 1, strTempLine.length());
                idTempDataPair.m_bHasLabels = true;
            }else{
                idTempDataPair.m_strDataName = strTempLine;
                idTempDataPair.m_strLabelName = "";
                idTempDataPair.m_bHasLabels = false;
            }
            
            strDataName = ssstrSrcPath + idTempDataPair.m_strDataName;
            strLabelName = ssstrDstPath + idTempDataPair.m_strLabelName;
            matDataMat = imread(strDataName, -1);
            matLabelMat = imread(strLabelName, -1);
            
            if(idTempDataPair.m_bHasLabels){
                if(matDataMat.empty()){
                    cerr << "File name:" << strDataName << " load failed!" << endl;
                    exit(1);
                }
                if(matLabelMat.empty()){
                    cerr << "File name:" << strLabelName << " load failed!" << endl;
                    exit(1);
                }
                
                IntersectionAndUnion(matDataMat, matLabelMat, vstrClsName, vmatIntersectionMat, vmatUnionMat);
                PixelAccuracy(matDataMat, matLabelMat, lLabelNum, lCorrectNum);
                FigureIOU(vmatIntersectionMat, vmatUnionMat, vdInterScore, vdUnionScore);
                
                //cout << strDataName << endl;
                //cout << strLabelName << endl;
                //cout << vdIOU.size() <<endl;
                /*dTotalAccuracy += dAccuracy;
                for(int l=0; l < vdIOU.size(); l++){
                    vdTotalIOU[l] += vdIOU[l];
                }*/     
            }
            nFile++;
        }
    }else{
        cerr << "Evaluation labels' nameList file open failed!" << endl;
        exit(1);
    }
    if(nFile){
        cout << "======= Summary IoU =======" << endl;
        for(int n = 0; n < vstrClsName.size(); n++){
            vdTotalIOU[n] = vdInterScore[n]/(vdUnionScore[n]);
            dTotalIOU += vdTotalIOU[n];
            cout << n << '\t' << vstrClsName[n] << ": " << vdTotalIOU[n] << endl;        
        }
        dTotalIOU /= vstrClsName.size(); 
        cout << "Mean IoU over 21 classes:" << dTotalIOU << endl;
        dTotalAccuracy = static_cast<double>(lCorrectNum) / lLabelNum;
        cout << "Pixel-wise Accuracy:" << dTotalAccuracy*100 << '\%' << endl;
        //cout << lCorrectNum << endl;
        //cout << lLabelNum << endl;
    }

    
    return 0;
}


void ParseClassName(const string&ssstrClsName, vector<string>&vstrClsName){
    ifstream isClsName(ssstrClsName.c_str());
    
    if(!isClsName.is_open()){
        cerr << "Class Name List file open failed!" << endl;
        exit(1);
    }
    
    if(vstrClsName.size()){
        vstrClsName.empty();
    }
    string strTempClsName;
    while(getline(isClsName, strTempClsName)){
        vstrClsName.push_back(strTempClsName);
        //cout << strTempClsName << endl;
    }
}

void IntersectionAndUnion(const Mat&matDataMat, const Mat&matLabelMat, const vector<string> vstrClsName, vector<Mat>&matIntersectionMat, vector<Mat>&matUnionMat){
    if(matDataMat.empty()){
        cerr << "IntersectionAndUnion matDataMat is empty!" << endl;
        exit(1);
    }
    
    if(matLabelMat.empty()){
        cerr << "IntersectionAndUnion matLabelMat is empty!" << endl;
        exit(1);
    }
    
    if(matDataMat.size() != matLabelMat.size()){
        cerr << "Data and label image don't have same size!" << endl;
        exit(1);
    }
    
    int nmatWidth = matDataMat.cols;
    int nmatHeight = matDataMat.rows;
    int nClsNum = vstrClsName.size();
    
    
    int r, c;
    const uchar *pDataMat = matDataMat.data;
    const uchar *pLabelMat = matLabelMat.data;
    uchar *pIntersectionMat, *pUnionMat;
    int n;
    int nTempR;
    matIntersectionMat.resize(nClsNum);
    matUnionMat.resize(nClsNum);
    for(n = 0;  n < nClsNum; n++){
        matIntersectionMat[n] = Mat::zeros(nmatHeight, nmatWidth, CV_8UC1);
        matUnionMat[n] = Mat::zeros(nmatHeight, nmatWidth, CV_8UC1);
        pIntersectionMat = matIntersectionMat[n].data;
        pUnionMat = matUnionMat[n].data;
        for(r = 0; r < nmatHeight; r++){
            nTempR = r * nmatWidth;
            for(c = 0; c < nmatWidth; c++){
                if(pDataMat[nTempR + c] == pLabelMat[nTempR + c] && n == pLabelMat[nTempR + c]){
                    pIntersectionMat[nTempR + c] = 1;
                }
                if(n == pLabelMat[nTempR + c] || n == pDataMat[nTempR + c]){
                    pUnionMat[nTempR + c] = 1;
                }
            }
        }
    }    
}

void PixelAccuracy(const Mat&matDataMat, const Mat&matLabelMat, long& lLabelNum, long& lCorrectNum){
    if(matDataMat.empty()){
        cerr << "PixelAccuracy matDataMat is empty!" << endl;
        exit(1);
    }
    
    if(matLabelMat.empty()){
        cerr << "PixelAccuracy matLabelMat is empty!" << endl;
        exit(1);
    }
    if(matDataMat.size() != matLabelMat.size()){
        cerr << "Data and label image don't have same size!" << endl;
        exit(1);
    }
    
    const uchar* pmatDataMat = matDataMat.data;
    const uchar* pmatLabelMat = matLabelMat.data;
    int nmatWidth = matDataMat.cols;
    int nmatHeight = matDataMat.rows;
    
    int r, c;
    int nr;
    int nLabelSum = 0;
    int nCorrectSum = 0;
    for(r = 0; r < matDataMat.rows; r++){
        nr = r * nmatWidth;
        for(c = 0; c < matDataMat.cols; c++){
            if(pmatDataMat[nr + c]){
                nLabelSum++;
            }
            if(pmatLabelMat[nr + c] == pmatDataMat[nr + c] && pmatDataMat[nr + c]){
                nCorrectSum++;
            }
        }        
    }
    lLabelNum += static_cast<long>(nLabelSum);
    lCorrectNum += static_cast<long>(nCorrectSum);
    /*if(nLabelSum)
        *pdAccuracy = static_cast<double>(nCorrectSum) / nLabelSum;
    else
        *pdAccuracy = 0.0;
    */
    
}

void FigureIOU(const vector<Mat>&vmatIntersectionMat, const vector<Mat>&vmatUnionMat, vector<double>&vdInterScore, vector<double>&vdUnionScore){
    if(vmatIntersectionMat.size() != vmatUnionMat.size()){
        cerr << "FigureIOU fuction input data size is not same!" << endl;
        exit(-1);
    }
    Scalar scSumInterMat, scSumUnionMat;
    vdInterScore.resize(vmatIntersectionMat.size(), 0.0f);
    vdUnionScore.resize(vmatIntersectionMat.size(), 0.0f);
    
    for(int n = 0; n < vmatIntersectionMat.size(); n++){
        scSumInterMat = sum(vmatIntersectionMat[n]);
        scSumUnionMat = sum(vmatUnionMat[n]);
        //cout << scSumInterMat << endl;
        //cout << scSumUnionMat << endl;
        if(scSumUnionMat[0] != 0){
            vdInterScore[n] += scSumInterMat[0];
            vdUnionScore[n] += scSumUnionMat[0];
            vnIOUSum[n]++;
        }
    }
}
