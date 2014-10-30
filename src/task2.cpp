#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "methods.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

/**
@mainpage Simple implementation of SVM classifier for images.
The program is able to train a model using some example pictures and use it to classify other images.
The program may also work with sse intrinsics for faster results.
*/

/**
@file task2.cpp

*/

///TDataSet - vector containing pairs of pointers to BMP images and corresponding labels
typedef vector<pair<BMP*, int> > TDataSet;
///TFileList - vector containing pairs of image paths and corresponding labels
typedef vector<pair<string, int> > TFileList;
///TFeatures - vector containing pairs of image features and corresponding labels
typedef vector<pair<vector<float>, int> > TFeatures;



/**
@function LoadFileList
Loads list of files and its labels from data_file and stores it in file_list
@param data_file is a string equal to path to the file that contains image paths and corresponding labels
@param file_list is a pointer to (@ref TFileList) that will contain pairs of image paths and corresponding labels
*/
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
        string data_path = data_file.substr(0,char_idx+1);

        while(!stream.eof() && !stream.fail()) {
            stream >> filename >> label;
            if (filename.size())
                file_list->push_back(make_pair(data_path + filename, label));
        }

        stream.close();
    }
/**
@function LoadImages 
Load images included in file_list and store them in data_set
@param file_list is (@ref TFileList) that contains pairs of image paths and corresponding labels
@param data_set is (@ref TDataSet) that will store loaded images and corresponding labels
*/

void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

/**
@function SavePredictions
Writes predicted labels, contained in labels, that correspond to image paths stored in file_list.
The output file path is equal to prediction_file.
@param file_list is a (@ref TFileList) that contains
*/
void SavePredictions(const TFileList& file_list,
   const TLabels& labels, 
   const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

/**
@function ExtractFeatures
Extract features form given dataset
@param data_set is a (@ref TDataSet) that contains loaded images and corresponding labels
@param features is a (@ref TFeatures) that will store the extracted features
@param useSse is a bool that specifies whether sse  intrinsics will be used
*/
void ExtractFeatures(const TDataSet& data_set, TFeatures* features, bool useSse) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        std::vector<float> result;
        Image gray = ImgToGrayscale(data_set[image_idx].first);
        Image hor(gray.n_rows, gray.n_cols);
        Image vert(gray.n_rows, gray.n_cols);
        ApplySobel(gray, hor, vert, useSse);
        floatImage magn = GetMagnitude(hor, vert, useSse);

        GetDescriptor(hor, vert, magn, result);

        //uncomment to run full tests
        /*uint halfRows = hor.n_rows >> 1;
        uint halfCols = hor.n_cols >> 1;

        GetDescriptor(   hor.submatrix(0, 0, halfRows, halfCols), 
                        vert.submatrix(0, 0, halfRows, halfCols),  
                        magn.submatrix(0, 0, halfRows, halfCols), 
                        result);

        GetDescriptor(   hor.submatrix(0, halfCols, halfRows, halfCols), 
                        vert.submatrix(0, halfCols, halfRows, halfCols), 
                        magn.submatrix(0, halfCols, halfRows, halfCols),
                        result);

        GetDescriptor(   hor.submatrix(halfRows, 0, halfRows, halfCols), 
                        vert.submatrix(halfRows, 0, halfRows, halfCols),
                        magn.submatrix(halfRows, 0, halfRows, halfCols),
                        result);

        GetDescriptor(   hor.submatrix(halfRows, halfCols, halfRows, halfCols), 
                        vert.submatrix(halfRows, halfCols, halfRows, halfCols), 
                        magn.submatrix(halfRows, halfCols, halfRows, halfCols), 
                        result);

        result = ApplyHIKernel(result);

        GetColors(data_set[image_idx].first, result);*/

        features->push_back(std::make_pair(result, data_set[image_idx].second));
    }
}

/**
@function ClearDataset
Free dataset resources
@param data_set is a (@ref TDataSet) which resources will be freed
*/
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}   

/**
@function TrainClassifier
Trains the SVM classifier 
@param data_file is a string that specifies the path to the file that contains images` names and corresponding labels
@param model_file is a string that specifies the path to the file that will store the model
@param useSse is a bool that specifies whether sse  intrinsics will be used
*/
void TrainClassifier(const string& data_file, const string& model_file, bool useSse) {
    //data_file == file with images` names and labels
    //model_file == output_file

        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, useSse);
        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**
@function PredictData
Classifies images using the model_file 
@param data_file is a string that specifies the path to the file that contains images` names
@param model_file is a string that specifies the path to the file that contains the model
@param useSse is a bool that specifies whether sse intrinsics will be used
*/
void PredictData(const string& data_file,
   const string& model_file,
   const string& prediction_file, bool useSse) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, useSse);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**
@function main
The main function
*/
int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    cmd.defineOption("sse", "Use sse");
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    bool useSse = cmd.foundOption("sse");
    if (useSse) {
        std::cout << "Using sse" << std::endl;
    }
    else {
        std::cout << "Not using sse" << std::endl;   
    }

        // If we need to train classifier

    if (train)
        TrainClassifier(data_file, model_file, useSse);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file, useSse);
    }
}