// mnist_test.cpp
// Testing-only MNIST digit classifier (C++17)

#include <bits/stdc++.h>
using namespace std;

/* ---------- Types ---------- */
using Vec = vector<double>;
using Mat = vector<Vec>;

/* ---------- Utility ---------- */
Vec softmax(const Vec &z) {
    double maxv = *max_element(z.begin(), z.end());
    Vec ex(z.size());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); i++) {
        ex[i] = exp(z[i] - maxv);
        sum += ex[i];
    }
    for (double &v : ex) v /= sum;
    return ex;
}

int argmax(const Vec &v) {
    return max_element(v.begin(), v.end()) - v.begin();
}

/* ---------- IDX Loader ---------- */
int read_int32(ifstream &ifs) {
    unsigned char b[4];
    ifs.read((char*)b, 4);
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

struct Dataset {
    Mat images;
    vector<int> labels;
    int n, dim;

    bool load_images(const string &file) {
        ifstream ifs(file, ios::binary);
        if (!ifs) return false;
        read_int32(ifs);
        n = read_int32(ifs);
        int r = read_int32(ifs);
        int c = read_int32(ifs);
        dim = r * c;
        images.assign(n, Vec(dim));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < dim; j++) {
                unsigned char p;
                ifs.read((char*)&p, 1);
                images[i][j] = p / 255.0;
            }
        return true;
    }

    bool load_labels(const string &file) {
        ifstream ifs(file, ios::binary);
        if (!ifs) return false;
        read_int32(ifs);
        int cnt = read_int32(ifs);
        labels.resize(cnt);
        for (int i = 0; i < cnt; i++) {
            unsigned char l;
            ifs.read((char*)&l, 1);
            labels[i] = l;
        }
        return true;
    }
};

/* ---------- Abstract Layer ---------- */
struct Layer {
    virtual Vec forward(const Vec&) = 0;
    virtual bool load(ifstream&) = 0;
    virtual ~Layer() {}
};

/* ---------- Dense Layer ---------- */
struct Dense : Layer {
    int in_dim, out_dim;
    Mat W;
    Vec b;

    Vec forward(const Vec &x) override {
        Vec out(out_dim, 0.0);
        for (int i = 0; i < out_dim; i++) {
            out[i] = b[i];
            for (int j = 0; j < in_dim; j++)
                out[i] += W[i][j] * x[j];
        }
        return out;
    }

    bool load(ifstream &ifs) override {
        ifs >> in_dim >> out_dim;
        if (ifs.fail()) {
            cout << "Failed to read dimensions for Dense layer\n";
            return false;
        }
        W.assign(out_dim, Vec(in_dim));
        for (int i = 0; i < out_dim; i++)
            for (int j = 0; j < in_dim; j++) {
                ifs >> W[i][j];
                if (ifs.fail()) {
                    cout << "Failed to read weights for Dense layer\n";
                    return false;
                }
            }
        b.resize(out_dim);
        for (int i = 0; i < out_dim; i++) {
            ifs >> b[i];
            if (ifs.fail()) {
                cout << "Failed to read biases for Dense layer\n";
                return false;
            }
        }
        return true;
    }
};

/* ---------- ReLU Layer ---------- */
struct ReLU : Layer {
    Vec forward(const Vec &x) override {
        Vec out = x;
        for (double &v : out) v = max(0.0, v);
        return out;
    }
    bool load(ifstream&) override { return true; }
};

/* ---------- Neural Network ---------- */
struct NeuralNetwork {
    vector<Layer*> layers;

    Vec forward(const Vec &x) {
        Vec out = x;
        for (auto L : layers)
            out = L->forward(out);
        return out;
    }

    int predict(const Vec &x) {
        Vec prob = softmax(forward(x));
        return argmax(prob);
    }

    void load(const string &file) {
        ifstream ifs(file);
        if (!ifs) {
            cout << "Failed to open model file " << file << "\n";
            return;
        }
        int n; ifs >> n;
        if (ifs.fail()) {
            cout << "Failed to read number of layers from model file\n";
            return;
        }
        layers.clear();
        for (int i = 0; i < n; i++) {
            Layer* layer;
            if (i == 1) layer = new ReLU();
            else layer = new Dense();
            if (!layer->load(ifs)) {
                cout << "Failed to load layer " << i << "\n";
                delete layer;
                return;
            }
            layers.push_back(layer);
        }
    }

    ~NeuralNetwork() {
        for (auto L : layers) delete L;
    }
};

/* ---------- MAIN TESTING ---------- */
int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Usage: ./mnist_test <model> <test-images> <test-labels>\n";
        return 1;
    }

    Dataset test;
    if (!test.load_images(argv[2])) {
        cout << "Failed to load images from " << argv[2] << "\n";
        return 1;
    }
    if (!test.load_labels(argv[3])) {
        cout << "Failed to load labels from " << argv[3] << "\n";
        return 1;
    }

    NeuralNetwork net;
    net.load(argv[1]);

    /* ---- STEP 5: Single Image + Confidence ---- */
    int idx = 0;
    Vec logits = net.forward(test.images[idx]);
    Vec prob = softmax(logits);

    cout << "STEP 5: Confidence Output\n";
    cout << "Actual Label: " << test.labels[idx] << "\n";
    cout << "Predicted Label: " << argmax(prob) << "\n";
    for (int i = 0; i < 10; i++)
        cout << "Digit " << i << " -> " << prob[i] << "\n";

    /* ---- STEP 6: Find Incorrect Predictions ---- */
    cout << "\nSTEP 6: First Incorrect Predictions\n";
    int shown = 0;
    for (int i = 0; i < test.n && shown < 5; i++) {
        int p = net.predict(test.images[i]);
        if (p != test.labels[i]) {
            cout << "Index " << i
                 << " Actual: " << test.labels[i]
                 << " Predicted: " << p << "\n";
            shown++;
        }
    }

    /* ---- Overall Accuracy ---- */
    int correct = 0;
    for (int i = 0; i < test.n; i++)
        if (net.predict(test.images[i]) == test.labels[i])
            correct++;

    cout << "\nFinal Test Accuracy: "
         << (100.0 * correct / test.n) << "%\n";

    return 0;
}
