#include <bits/stdc++.h>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

static inline double uniform_rand() {
    return (double)rand() / RAND_MAX;
}

Mat zeros(int r, int c) {
    return Mat(r, Vec(c, 0.0));
}

Vec zerosV(int n) { return Vec(n, 0.0); }

Mat random_mat(int r, int c, double scale=0.01) {
    Mat m(r, Vec(c));
    for(int i=0;i<r;i++) for(int j=0;j<c;j++) m[i][j] = (uniform_rand()*2-1)*scale;
    return m;
}

Vec softmax(const Vec &z) {
    double maxv = *max_element(z.begin(), z.end());
    Vec ex(z.size());
    double s=0;
    for(size_t i=0;i<z.size();++i){ ex[i]=exp(z[i]-maxv); s+=ex[i]; }
    for(size_t i=0;i<z.size();++i) ex[i]/=s;
    return ex;
}

int argmax(const Vec &v) {
    return max_element(v.begin(), v.end()) - v.begin();
}

static int read_int32(ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read((char*)bytes, 4);
    return (int)((bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3]);
}

struct Dataset {
    Mat images; // each row flattened => 28*28 = 784
    vector<int> labels;
    int n, dim;
    Dataset(): n(0), dim(0) {}
    bool load_images(const string &fname) {
        ifstream ifs(fname, ios::binary);
        if(!ifs){ cerr<<"Cannot open "<<fname<<"\n"; return false;}
        int magic = read_int32(ifs);
        int num = read_int32(ifs);
        int rows = read_int32(ifs);
        int cols = read_int32(ifs);
        dim = rows*cols;
        images.resize(num, Vec(dim));
        for(int i=0;i<num;i++){
            for(int j=0;j<dim;j++){
                unsigned char pix; ifs.read((char*)&pix,1);
                images[i][j] = ((double)pix)/255.0;
            }
        }
        n = num;
        return true;
    }
    bool load_labels(const string &fname) {
        ifstream ifs(fname, ios::binary);
        if(!ifs){ cerr<<"Cannot open "<<fname<<"\n"; return false;}
        int magic = read_int32(ifs);
        int num = read_int32(ifs);
        labels.resize(num);
        for(int i=0;i<num;i++){
            unsigned char lab; ifs.read((char*)&lab,1);
            labels[i] = (int)lab;
        }
        return true;
    }
};

struct Layer {
    virtual Vec forward(const Vec &input) = 0;
    virtual Vec backward(const Vec &grad_output, double lr) = 0;
    virtual void save(ofstream &ofs) = 0;
    virtual void load(ifstream &ifs) = 0;
    virtual ~Layer() {}
};

struct Dense : Layer {
private:
    int in_dim, out_dim;
    Mat W;
    Vec b;
    Vec last_input;
    Mat gradW_acc;
public:
    Dense(int in_dim_, int out_dim_, double init_scale=0.01) : in_dim(in_dim_), out_dim(out_dim_) {
        W = random_mat(out_dim, in_dim, init_scale);
        b = zerosV(out_dim);
        gradW_acc = zeros(out_dim, in_dim);
    }

    Vec forward(const Vec &input) override {
        last_input = input;
        Vec out(out_dim, 0.0);
        for(int i=0;i<out_dim;i++){
            double s = b[i];
            for(int j=0;j<in_dim;j++){
                s += W[i][j] * input[j];
            }
            out[i] = s;
        }
        return out;
    }

    Vec backward(const Vec &grad_output, double lr) override {
        Vec grad_input(in_dim, 0.0);
        // update weights with simple SGD
        for(int i=0;i<out_dim;i++){
            for(int j=0;j<in_dim;j++){
                double grad_w = grad_output[i] * last_input[j];
                W[i][j] -= lr * grad_w;
            }
            b[i] -= lr * grad_output[i];
        }
        // compute grad_input
        for(int j=0;j<in_dim;j++){
            double s=0;
            for(int i=0;i<out_dim;i++) s += W[i][j] * grad_output[i];
            grad_input[j] = s;
        }
        return grad_input;
    }

    void save(ofstream &ofs) override {
        ofs << in_dim << " " << out_dim << "\n";
        for(int i=0;i<out_dim;i++){
            for(int j=0;j<in_dim;j++) ofs << W[i][j] << " ";
            ofs << "\n";
        }
        for(int i=0;i<out_dim;i++) ofs << b[i] << " ";
        ofs << "\n";
    }
    void load(ifstream &ifs) override {
        int id, od; ifs >> id >> od;
        in_dim = id; out_dim = od;
        W = zeros(out_dim, in_dim);
        for(int i=0;i<out_dim;i++) for(int j=0;j<in_dim;j++) ifs >> W[i][j];
        b = zerosV(out_dim);
        for(int i=0;i<out_dim;i++) ifs >> b[i];
    }
};

struct ReLU : Layer {
private:
    Vec last_input;
public:
    Vec forward(const Vec &input) override {
        last_input = input;
        Vec out(input.size());
        for(size_t i=0;i<input.size();++i) out[i] = max(0.0, input[i]);
        return out;
    }
    Vec backward(const Vec &grad_output, double /*lr*/) override {
        Vec grad_input(grad_output.size());
        for(size_t i=0;i<grad_output.size();++i) grad_input[i] = (last_input[i] > 0) ? grad_output[i] : 0.0;
        return grad_input;
    }
    void save(ofstream &/*ofs*/) override { /* nothing to save */ }
    void load(ifstream &/*ifs*/) override { /* nothing to load */ }
};

struct SoftmaxCrossEntropy {
    // forward returns loss and stores last softmax/prob
    Vec last_prob;
    double forward_and_loss(const Vec &logits, int label) {
        last_prob = softmax(logits);
        double loss = -log(max(1e-12, last_prob[label]));
        return loss;
    }
    Vec backward(int label) {
        Vec grad = last_prob;
        grad[label] -= 1.0; // derivative of cross entropy with softmax
        return grad;
    }
};

struct NeuralNetwork {
    vector<Layer*> layers;
    SoftmaxCrossEntropy loss;

    Vec forward(const Vec &x) {
        Vec out = x;
        for(auto L : layers) out = L->forward(out);
        return out;
    }

    double train_step(const Vec &x, int y, double lr) {
        Vec logits = forward(x);
        double l = loss.forward_and_loss(logits, y);
        Vec grad = loss.backward(y);
        for(int i = (int)layers.size()-1; i>=0; --i) {
            grad = layers[i]->backward(grad, lr);
        }
        return l;
    }

    int predict(const Vec &x) {
        Vec logits = forward(x);
        Vec prob = softmax(logits);
        return argmax(prob);
    }

    void save(const string &fname) {
        ofstream ofs(fname);
        ofs << layers.size() << "\n";
        for(auto L : layers) L->save(ofs);
        ofs.close();
    }
    void load(const string &fname) {
        ifstream ifs(fname);
        int n; ifs >> n;
        for(auto L : layers) L->load(ifs);
        ifs.close();
    }

    ~NeuralNetwork() {
        for(auto L : layers) delete L;
    }
};

double evaluate_accuracy(NeuralNetwork &net, Dataset &ds, int max_samples=10000) {
    int correct=0;
    int total = min(ds.n, max_samples);
    for(int i=0;i<total;i++){
        int p = net.predict(ds.images[i]);
        if(p == ds.labels[i]) ++correct;
    }
    return 100.0 * ((double)correct / total);
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    srand(1234);
    if(argc < 5) {
        cout << "Usage: " << argv[0] << " <train-images> <train-labels> <test-images> <test-labels>\n";
        cout << "Example: ./mnist_nn train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte\n";
        return 1;
    }
    string train_images = argv[1];
    string train_labels = argv[2];
    string test_images  = argv[3];
    string test_labels  = argv[4];

    Dataset train, test;
    if(!train.load_images(train_images) || !train.load_labels(train_labels)) return 2;
    if(!test.load_images(test_images) || !test.load_labels(test_labels)) return 2;

    cout << "Loaded train samples: " << train.n << " dim=" << train.dim << "\n";
    cout << "Loaded test samples: " << test.n << " dim=" << test.dim << "\n";

    // build network: 784 -> 128 -> 10
    NeuralNetwork net;
    net.layers.push_back(new Dense(train.dim, 128, 0.01));
    net.layers.push_back(new ReLU());
    net.layers.push_back(new Dense(128, 10, 0.01));
    // training hyperparams
    int epochs = 5;
    int batch_size = 10;
    double lr = 0.001;

    int steps = 0;
    for(int e=0;e<epochs;e++){
        double epoch_loss = 0.0;
        // shuffle indices
        vector<int> idx(train.n);
        iota(idx.begin(), idx.end(), 0);
        random_shuffle(idx.begin(), idx.end());
        for(int i=0;i<train.n;i++){
            Vec &x = train.images[idx[i]];
            int y = train.labels[idx[i]];
            double l = net.train_step(x, y, lr);
            epoch_loss += l;
            if(++steps % 500 == 0) {
                double acc = evaluate_accuracy(net, test, 1000);
                cout << "[epoch " << e << "] step " << steps << " recent loss " << l << " test_acc@1k " << acc << "%\n";
            }
        }
        cout << "Epoch " << e << " avg loss: " << (epoch_loss / train.n) << "\n";
    }

    double final_acc = evaluate_accuracy(net, test);
    cout << "Final test accuracy: " << final_acc << "%\n";

    // save model
    net.save("mnist_model.txt");
    cout << "Model saved to mnist_model.txt\n";
    return 0;
}
