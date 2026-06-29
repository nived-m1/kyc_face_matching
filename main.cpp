#define STB_IMAGE_IMPLEMENTATION  
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

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

img resizeFaceCrop(img input, int target_dim, int offsetX, int offsetY) {
    img output;
    output.w = target_dim;
    output.h = target_dim;
    output.c = input.c;
    output.data = new unsigned char[target_dim * target_dim];

    int minDim = std::min(input.w, input.h);
    int cropSize = (int)(minDim * 0.55f); 
    int startX = (input.w - cropSize) / 2 + offsetX;
    int startY = (input.h - cropSize) / 4 + offsetY; 

    for (int y = 0; y < target_dim; y++) {
        for (int x = 0; x < target_dim; x++) {
            int cropX = (x * cropSize) / target_dim;
            int cropY = (y * cropSize) / target_dim;
            
            int srcX = std::max(0, std::min(input.w - 1, startX + cropX));
            int srcY = std::max(0, std::min(input.h - 1, startY + cropY));
            
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
    
    int cellSize = 16; 
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
    if (argc < 3) {
        return 1;
    }

    string selfieFile = argv[1];
    string idFile = argv[2];

    img rawSelfie = loadImage(selfieFile);
    img rawId = loadImage(idFile);

    if (rawSelfie.w == 0 || rawId.w == 0) {
        if(rawSelfie.data) stbi_image_free(rawSelfie.data);
        if(rawId.data) stbi_image_free(rawId.data);
        return 1;
    }

    img idDocResized = resizeFaceCrop(rawId, 256, 0, 0);
    img idDoc = normalizeImage(idDocResized);
    vector<float> hog2 = doHOG(idDoc);

    float bestMatchPercentage = 0.0f;
    float bestHogScore = 0.0f;
    float bestMseScore = 0.0f;

    int searchStep = rawSelfie.w / 20; 
    if (searchStep == 0) searchStep = 10;
    int maxOffset = searchStep * 2;

    for (int dy = -maxOffset; dy <= maxOffset; dy += searchStep) {
        for (int dx = -maxOffset; dx <= maxOffset; dx += searchStep) {
            
            img selfieResized = resizeFaceCrop(rawSelfie, 256, dx, dy);
            img selfie = normalizeImage(selfieResized);

            float mse = getMSE(selfie, idDoc);
            vector<float> hog1 = doHOG(selfie);

            float dotProduct = 0.0f;
            float norm1 = 0.0f;
            float norm2 = 0.0f;
            int size = std::min(hog1.size(), hog2.size());
            
            for (int i = 0; i < size; i++) {
                dotProduct += (hog1[i] * hog2[i]);
                norm1 += (hog1[i] * hog1[i]);
                norm2 += (hog2[i] * hog2[i]);
            }

            float cosineSimilarity = dotProduct / (sqrt(norm1) * sqrt(norm2) + 1e-7f);

            float maxMseBoundary = 45000.0f; 
            float mseScore = std::max(0.0f, (1.0f - (mse / maxMseBoundary))) * 100.0f;
            float hogScore = std::max(0.0f, cosineSimilarity) * 100.0f;
            
            float matchPercentage = (mseScore * 0.15f) + (hogScore * 0.85f);

            if (matchPercentage > bestMatchPercentage) {
                bestMatchPercentage = matchPercentage;
                bestHogScore = hogScore;
                bestMseScore = mseScore;
            }

            delete[] selfieResized.data;
            delete[] selfie.data;
        }
    }

    cout << "\n=== KYC VERIFICATION REPORT ===" << endl;
    cout << "Best Alignment Component 1 (MSE): " << bestMseScore << "% Match" << endl;
    cout << "Best Alignment Component 2 (HOG): " << bestHogScore << "% Match" << endl;
    cout << "-------------------------------" << endl;
    cout << "Final Confidence : " << bestMatchPercentage << "%" << endl;
    
    if (bestMatchPercentage >= 75.0f) {
        cout << "DECISION         : [ APPROVED - MATCH FOUND ]" << endl;
    } else {
        cout << "DECISION         : [ REJECTED - NO MATCH ]" << endl;
    }
    cout << "===============================\n" << endl;

    delete[] idDocResized.data;
    delete[] idDoc.data;

    if(rawSelfie.data) stbi_image_free(rawSelfie.data);
    if(rawId.data) stbi_image_free(rawId.data);

    return 0;
}