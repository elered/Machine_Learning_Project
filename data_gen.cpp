#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "funzioni.h" 

using namespace std;

int main() {
    QuantumGenerator qgen;
    int n_samples_per_class = 5000;
    ofstream file("quantum_data_rich.csv");

    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file!" << endl;
        return 1;
    }

    for (int i = 0; i < 32; i++) file << "raw_f" << i << ",";
    file << "is_entangled,violates_bell,bell_value\n";

    cout << "Inizio generazione dati (32 feature + Bell target)..." << endl;

    cout << "Generazione stati separabili..." << endl;
    for(int i = 0; i < n_samples_per_class; i++) {
        Matrix4cd rho = qgen.generateSeparableDM();
        
        vector<double> features = qgen.getRawFeatures(rho);
        double bell_s = qgen.getBellValue(rho);

        for(double f : features) file << f << ",";
        
        file << 0 << ",";            
        file << 0 << ",";            
        file << bell_s << "\n";       
    }

    cout << "Generazione stati entangled..." << endl;
    int count_entangled = 0;
    while(count_entangled < n_samples_per_class) {
        Matrix4cd rho = qgen.generateRandomDM();

        if(qgen.isEntangled(rho)) {
            vector<double> features = qgen.getRawFeatures(rho);
            double bell_s = qgen.getBellValue(rho);
            int bell_violated = (bell_s > 2.0) ? 1 : 0;

            for(double f : features) file << f << ",";
            
            file << 1 << ",";             
            file << bell_violated << ",";   
            file << bell_s << "\n";         
            
            count_entangled++;
            if(count_entangled % 1000 == 0) cout << "Entangled generati: " << count_entangled << "..." << endl;
        }
    }

    file.close();
    cout << "Dataset pronto: 'quantum_data_rich.csv'" << endl;
    cout << "Struttura: 32 Feature | Entangled_Label | Bell_Label | Bell_Value" << endl;

    return 0;
}