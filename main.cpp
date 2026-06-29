#define STB_IMAGE_IMPLEMENTATION  
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

#include <iostream>
#include <vector>      
#include <cmath> 
#include <string>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

struct img {
    int w, h, c;
    unsigned char* data;
};

img loadImage(const string& path) {
    img image;
    image.data = stbi_load(path.c_str(), &image.w, &image.h, &image.c, 1);
    if (!image.data) {
        image.w = 0;
    }
    image.c = 1; 
    return image;
}

img resizeCenterCrop(img input, int target_dim) {
    img output;
    output.w = target_dim;
    output.h = target_dim;
    output.c = input.c;
    output.data = new unsigned char[target_dim * target_dim];

    int minDim = std::min(input.w, input.h);
    int startX = (input.w - minDim) / 2;
    int startY = (input.h - minDim) / 2;

    for (int y = 0; y < target_dim; y++) {
        for (int x = 0; x < target_dim; x++) {
            int cropX = (x * minDim) / target_dim;
            int cropY = (y * minDim) / target_dim;
            
            int srcX = startX + cropX;
            int srcY = startY + cropY;
            
            output.data[y * target_dim + x] = input.data[srcY * input.w + srcX];
        }
    }
    return output;
}

img normalizeImage(img input) {
    img output;
    output.w = input.w;
    output.h = input.h;
    output.c = input.c;
    output.data = new unsigned char[input.w * input.h];
    
    int minVal = 255, maxVal = 0;
    for (int i = 0; i < input.w * input.h; i++) {
        if (input.data[i] < minVal) minVal = input.data[i];
        if (input.data[i] > maxVal) maxVal = input.data[i];
    }
    
    int range = maxVal - minVal;
    if (range == 0) range = 1;
    
    for (int i = 0; i < input.w * input.h; i++) {
        float normalized = (float)(input.data[i] - minVal) / range * 255.0f;
        output.data[i] = (unsigned char)normalized;
    }
    
    return output;
}

float getMSE(img i1, img i2) {
    long long sumSqDiff = 0;
    int totalPixels = i1.w * i1.h;
    
    for (int i = 0; i < totalPixels; i++) {
        int diff = i1.data[i] - i2.data[i];
        sumSqDiff += (diff * diff);
    }
    
    return (float)sumSqDiff / totalPixels;
}

vector<float> doHOG(img input) {
    int width = input.w;
    int height = input.h;
    
    vector<float> gradX(width * height, 0.0f);
    vector<float> gradY(width * height, 0.0f);
    vector<float> magnitude(width * height, 0.0f);
    vector<float> orientation(width * height, 0.0f);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = input.data[y * width + (x + 1)] - input.data[y * width + (x - 1)];
            float gy = input.data[(y + 1) * width + x] - input.data[(y - 1) * width + x];
            
            gradX[y * width + x] = gx;
            gradY[y * width + x] = gy;
            magnitude[y * width + x] = sqrt(gx * gx + gy * gy);
            
            float angle = atan2(gy, gx) * 180.0f / M_PI;
            if (angle < 0) angle += 180.0f;
            if (angle >= 180.0f) angle -= 180.0f;
            orientation[y * width + x] = angle;
        }
    }
    
    int cellSize = 8;
    int numBins = 9;
    int cellsX = width / cellSize;
    int cellsY = height / cellSize;
    
    vector<vector<vector<float>>> cellHistograms(cellsY, vector<vector<float>>(cellsX, vector<float>(numBins, 0.0f)));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cellIdxX = x / cellSize;
            int cellIdxY = y / cellSize;
            
            if (cellIdxX >= cellsX || cellIdxY >= cellsY) continue;
            
            float mag = magnitude[y * width + x];
            float angle = orientation[y * width + x];
            
            float binWidth = 20.0f;
            float binBase = angle / binWidth;
            int bin1 = (int)floor(binBase) % numBins;
            int bin2 = (bin1 + 1) % numBins;
            
            float weight2 = binBase - floor(binBase);
            float weight1 = 1.0f - weight2;
            
            cellHistograms[cellIdxY][cellIdxX][bin1] += mag * weight1;
            cellHistograms[cellIdxY][cellIdxX][bin2] += mag * weight2;
        }
    }
    
    vector<float> hogFeatures;
    
    for (int y = 0; y < cellsY - 1; y++) {
        for (int x = 0; x < cellsX - 1; x++) {
            vector<float> blockVec;
            
            for (int by = 0; by < 2; by++) {
                for (int bx = 0; bx < 2; bx++) {
                    for (int bin = 0; bin < numBins; bin++) {
                        blockVec.push_back(cellHistograms[y + by][x + bx][bin]);
                    }
                }
            }
            
            float normSum = 0.0f;
            for (float val : blockVec) normSum += val * val;
            float norm = sqrt(normSum + 1e-5f);
            
            for (float val : blockVec) {
                hogFeatures.push_back(val / norm);
            }
        }
    }
    
    return hogFeatures;
}

int main(int argc, char* argv[]) {
    cout << "=== KYC Biometric Verification Engine ===" << endl;

    if (argc < 3) {
        cout << "\nError: Missing file arguments." << endl;
        cout << "Usage: ./test_app <selfie_image> <passport_image>" << endl;
        return 1;
    }

    string selfieFile = argv[1];
    string idFile = argv[2];

    cout << "Loading: " << selfieFile << " and " << idFile << "..." << endl;

    img rawSelfie = loadImage(selfieFile);
    img rawId = loadImage(idFile);

    if (rawSelfie.w == 0 || rawId.w == 0) {
        cout << "Error: Execution halted. Verification assets missing or corrupt." << endl;
        if(rawSelfie.data) stbi_image_free(rawSelfie.data);
        if(rawId.data) stbi_image_free(rawId.data);
        return 1;
    }

    img selfieResized = resizeCenterCrop(rawSelfie, 256);
    img idDocResized = resizeCenterCrop(rawId, 256);

    img selfie = normalizeImage(selfieResized);
    img idDoc = normalizeImage(idDocResized);

    float mse = getMSE(selfie, idDoc);
    cout << "MSE: " << mse << endl;

    cout << "Doing HOG extraction..." << endl;
    vector<float> hog1 = doHOG(selfie);
    vector<float> hog2 = doHOG(idDoc);

    float diffSum = 0;
    int size = std::min(hog1.size(), hog2.size());
    
    for (int i = 0; i < size; i++) {
        float d = hog1[i] - hog2[i];
        diffSum += (d * d);
    }
    
    float hogDist = sqrt(diffSum);
    cout << "HOG Distance: " << hogDist << endl;

    float maxMseBoundary = 20000.0f;
    float maxHogBoundary = 60.0f;

    float mseScore = (1.0f - (mse / maxMseBoundary)) * 100.0f;
    float hogScore = (1.0f - (hogDist / maxHogBoundary)) * 100.0f;

    if (mseScore < 0.0f) mseScore = 0.0f;
    if (hogScore < 0.0f) hogScore = 0.0f;

    float matchPercentage = (mseScore * 0.1f) + (hogScore * 0.9f);

    cout << "---------------------------------------" << endl;
    cout << "Biometric Match Confidence: " << matchPercentage << "%" << endl;
    cout << "---------------------------------------" << endl;

    delete[] selfieResized.data;
    delete[] idDocResized.data;
    delete[] selfie.data;
    delete[] idDoc.data;

    if(rawSelfie.data) stbi_image_free(rawSelfie.data);
    if(rawId.data) stbi_image_free(rawId.data);

    return 0;
}