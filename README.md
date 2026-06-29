# KYC Face Matching

This is a lightweight C/C++ application built for KYC (Know Your Customer) identity verification. The program processes two facial images and compares their features to determine if they are a match.

## Functions:
To keep the project simple and minimize dependencies, it relies on the single-header `stb_image.h` library to load and parse the images instead of using heavy external frameworks.

For the core logic, the application works in a few steps:
* It scans the images using a sliding window technique.
* It applies HOG (Histogram of Oriented Gradients) and MSE(Mean Squared Error) to extract the distinct shapes and structures of the faces.
* It maps these features into vectors and compares them using both Cosine Similarity and Mean Squared Error (MSE).

By combining these metrics, the program calculates a confidence score to verify the identity.

## Getting Started

To run this on your own machine, you just need Git and a standard C/C++ compiler (like GCC or Clang).

1. Clone the repository:
```bash
git clone [https://github.com/nived-m1/kyc_face_matching.git](https://github.com/nived-m1/kyc_face_matching.git)
cd kyc_face_matching
