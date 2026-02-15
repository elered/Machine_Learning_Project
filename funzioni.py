import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import mutual_info_classif
import itertools
from sklearn.metrics import log_loss 
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import validation_curve
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import balanced_accuracy_score

def reconstruct_rho(row):
    raw_vals = row.filter(like='raw').values
    if len(raw_vals) < 32: 
        return None
    complex_vals = raw_vals[0::2] + 1j * raw_vals[1::2]
    return complex_vals.reshape((4, 4))

def calculate_entanglement_entropy(row):
    rho = reconstruct_rho(row)
    if rho is None: return np.nan
    
    # Traccia Parziale (A)
    rho_a = np.zeros((2, 2), dtype=complex)
    rho_a[0, 0] = rho[0, 0] + rho[1, 1]
    rho_a[0, 1] = rho[0, 2] + rho[1, 3]
    rho_a[1, 0] = rho[2, 0] + rho[3, 1]
    rho_a[1, 1] = rho[2, 2] + rho[3, 3]
    
    evs = np.clip(np.linalg.eigvalsh(rho_a), 1e-12, 1.0)
    return -np.sum(evs * np.log2(evs))

def calculate_purity(row):
    rho = reconstruct_rho(row)
    if rho is None: return np.nan

    return np.real(np.trace(rho @ rho))

def calculate_state_entropy(row):
    rho = reconstruct_rho(row)
    if rho is None: return np.nan

    evs = np.clip(np.linalg.eigvalsh(rho), 1e-12, 1.0)
    return -np.sum(evs * np.log2(evs))

def esegui_e_plotta_tutto(df, X_pca, target_col, nome_target, mappa_colore, ax_matrix, ax_text, C=1.0):
    if df[target_col].dtype == object:
        y = df[target_col].astype(str).str.contains('Entangled|Viola|1|True').astype(int)
    else:
        y = df[target_col].fillna(0).astype(int)
        
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(class_weight='balanced', max_iter=5000, C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = ['Separabile/Locale', 'Entangled/Viola Bell']
    
    # Matrice di Confusione
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=mappa_colore, cbar=False, ax=ax_matrix,
                xticklabels=labels, yticklabels=labels)
    
    acc = accuracy_score(y_test, y_pred)
    ax_matrix.set_title(f'{nome_target}\nAcc: {acc:.2%}', fontsize=12, fontweight='bold')
    ax_matrix.set_xlabel('Predetto')
    ax_matrix.set_ylabel('Reale')
    
    # Report di classificazione
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    ax_text.text(0.01, 0.5, report, {'fontsize': 10, 'fontfamily': 'monospace'}, va='center')
    ax_text.axis('off')

    return model

def studio_miglior_C(df, X_pca, target_col, nome_target):
    if df[target_col].dtype == object:
        y = df[target_col].astype(str).str.contains('Entangled|Viola|1|True').astype(int)
    else:
        y = df[target_col].fillna(0).astype(int)

    param_range = np.logspace(-4, 4, 10)
    train_scores, test_scores = validation_curve(
        LogisticRegression(class_weight='balanced', max_iter=5000), 
        X_pca, y, param_name="C", param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    
    # Medie e Deviazioni Standard
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    # Identificazione del valore ottimo
    best_idx = np.argmax(test_mean)
    best_C = param_range[best_idx]
    best_acc = test_mean[best_idx]

    # Risultati
    print(f"--- RISULTATI OTTIMIZZAZIONE: {nome_target} ---")
    print(f"Miglior Valore di C: {best_C:.4f}")
    print(f"Accuracy Massima (CV): {best_acc:.2%}\n")

    # Plot delle curve
    plt.figure(figsize=(10, 5))
    plt.semilogx(param_range, train_mean, label="Training Score", color="darkorange", marker='o')
    plt.semilogx(param_range, test_mean, label="Cross-Validation Score", color="navy", marker='s')

    plt.axvline(best_C, linestyle="--", color="red", alpha=0.6, label=f"Ottimo (C={best_C:.5f})")
    
    plt.title(f"Studio della Regolarizzazione: {nome_target}", fontsize=14)
    plt.xlabel("Parametro C (Scala Logaritmica)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()

    return best_C

def estrai_errori_singoli(X_pca, y):
    # Split dei dati
    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Training del modello
    model = LogisticRegression(class_weight='balanced', max_iter=2000)
    model.fit(X_tr, y_tr)
    
    # Predizione probabilità
    p_tr = model.predict_proba(X_tr)
    p_te = model.predict_proba(X_te)
    
    # Calcolo metriche
    bias_sq = log_loss(y_tr, p_tr)
    err_test = log_loss(y_te, p_te)
    varianza = max(0, err_test - bias_sq)
    
    return {
        'Bias_Quadrato': bias_sq,
        'Varianza': varianza,
        'Errore_Totale': err_test
    }

def esegui_svm_e_plotta(df, X_pca, target_col, nome_target, mappa_colore, ax_matrix, ax_text, parametri):

    if df[target_col].dtype == object:
        y = df[target_col].astype(str).str.contains('Entangled|Viola|1|True').astype(int)
    else:
        y = df[target_col].fillna(0).astype(int)
        
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', **parametri, class_weight='balanced', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    labels = ['Separabile/Locale', 'Entangled/Viola Bell']

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=mappa_colore, cbar=False, ax=ax_matrix,
                xticklabels=labels, yticklabels=labels)
    
    acc = accuracy_score(y_test, y_pred)
    ax_matrix.set_title(f'SVM Ottimizzata: {nome_target}\nAcc: {acc:.2%}', fontsize=14, fontweight='bold')
    ax_matrix.set_xlabel('Predetto')
    ax_matrix.set_ylabel('Reale')

    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    ax_text.text(0.01, 0.5, report, {'fontsize': 11, 'fontfamily': 'monospace'}, va='center')
    ax_text.axis('off')

    return model

def plot_confronto_permutation(model_ent, model_bell, df, X_pca, n_95):
    
    if df['is_entangled'].dtype == object:
        y_ent = df['is_entangled'].astype(str).str.contains('Entangled|1|True').astype(int)
    else:
        y_ent = df['is_entangled'].fillna(0).astype(int)
    
    # Split per Entanglement
    _, X_te_e, _, y_te_e = train_test_split(X_pca, y_ent, test_size=0.2, random_state=42)
    
    # Calcolo Importanza
    res_ent = permutation_importance(model_ent, X_te_e, y_te_e, n_repeats=10, random_state=42)
    imp_ent = res_ent.importances_mean

    if df['violates_bell'].dtype == object:
        y_bell = df['violates_bell'].astype(str).str.contains('Viola|1|True').astype(int)
    else:
        y_bell = df['violates_bell'].fillna(0).astype(int)
    
    # Split per Bell
    _, X_te_b, _, y_te_b = train_test_split(X_pca, y_bell, test_size=0.2, random_state=42)
    
    # Calcolo Importanza
    res_bell = permutation_importance(model_bell, X_te_b, y_te_b, n_repeats=10, random_state=42)
    imp_bell = res_bell.importances_mean

    # Plot
    pc_labels = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    x = np.arange(len(pc_labels))
    width = 0.35

    plt.figure(figsize=(15, 6))
    plt.bar(x - width/2, imp_ent, width, label='Importanza Entanglement', color='blue', alpha=0.7)
    plt.bar(x + width/2, imp_bell, width, label='Importanza Bell', color='red', alpha=0.7)

    plt.axvline(n_95 - 0.5, color='black', linestyle='--', label=f'Soglia 95% (n={n_95})')
    plt.title("Confronto Importanza Feature (Permutation Importance)", fontsize=14)
    plt.xlabel("Componenti Principali (PCA)")
    plt.ylabel("Diminuzione Accuracy")
    plt.xticks(x, pc_labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def calcola_metrics_bv_svm(X_std, y, max_comp, best_params):
    bias_sq, variance, total_err = [], [], []
    comp_range = range(1, max_comp + 1)
    
    for n in comp_range:
        pca_t = PCA(n_components=n)
        X_p = pca_t.fit_transform(X_std)
        X_tr, X_te, y_tr, y_te = train_test_split(X_p, y, test_size=0.2, random_state=42)
        
        # Addestramento SVM con i parametri trovati prima
        model = SVC(kernel='rbf', **best_params, probability=True, 
                    class_weight='balanced', random_state=42)
        model.fit(X_tr, y_tr)
        
        # Calcolo Log-Loss 
        p_tr = model.predict_proba(X_tr)
        p_te = model.predict_proba(X_te)
        
        err_tr = log_loss(y_tr, p_tr)
        err_te = log_loss(y_te, p_te)
        
        # 4. Salvataggio metriche
        bias_sq.append(err_tr)       
        total_err.append(err_te)    
        variance.append(max(0, err_te - err_tr)) 
        
    return comp_range, bias_sq, variance, total_err

def analizza_support_vectors(model, df_originale, X_pca, target_col):
    
    df_train, _, _, _ = train_test_split(df_originale, df_originale[target_col], 
                                         test_size=0.2, random_state=42)
    
    # Estraiamo i Support Vectors
    sv_indices = model.support_
    sv_states = df_train.iloc[sv_indices]

    tipo_analisi = "Entanglement (PPT)" if target_col == 'is_entangled' else "Non-Località (Bell)"
    
    plt.figure(figsize=(12, 6))

    sns.kdeplot(df_originale['bell_value'], label='Popolazione Totale', 
                fill=True, alpha=0.1, color='gray', bw_adjust=0.7)

    sns.kdeplot(sv_states['bell_value'], label='Support Vectors (Stati Critici)', 
                fill=True, alpha=0.5, color='teal', bw_adjust=0.7)
    
    plt.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Limite Località ($S=2$)')
    
    plt.title(f"Distribuzione Fisica degli Stati Critici per: {tipo_analisi}", fontsize=14)
    plt.xlabel("Parametro di Bell ($S$)", fontsize=12)
    plt.ylabel("Densità di probabilità", fontsize=12)
    
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    print(f"--- Insight per {tipo_analisi} ---")
    print(f"Numero di Support Vectors: {len(sv_indices)} ({len(sv_indices)/len(df_train):.1%})")
    print(f"Valore medio di S per gli Stati Critici: {sv_states['bell_value'].mean():.3f}")

def plot_svm_heatmap_v2(X, y, nome_target):

    scorer = make_scorer(balanced_accuracy_score)
    
    X_random, y_random = resample(X, y, n_samples=3000, random_state=42, stratify=y)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': ['scale', 0.1, 0.01, 0.001, 0.0001]
    }

    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), 
                        param_grid, cv=5, scoring=scorer, n_jobs=-1)
    grid.fit(X_random, y_random)
    
    results = pd.DataFrame(grid.cv_results_)
    pivot = results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt=".3f", 
                vmin=pivot.values.min(), vmax=pivot.values.max())
    
    plt.title(f"Mappa di Sensibilità (Balanced Accuracy): {nome_target}", fontsize=15)
    plt.show()

    return grid.best_params_

class QuantumMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.1, activation_fn=nn.SiLU):
        super(QuantumMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1)) 
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, X_val, y_val, criterion, optimizer, epochs=300, patience=20):
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    final_epoch = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y.view(-1))
            loss.backward()
            optimizer.step()
        
        # Validazione
        model.eval()
        with torch.no_grad():
            val_out = model(X_val).view(-1)
            val_loss = criterion(val_out, y_val.view(-1))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        final_epoch = epoch
        if counter >= patience:
            break
            
    return best_model_state, best_val_loss, final_epoch

def run_grid_search(input_dim, train_loader, X_test_t, y_test_t, y_test_raw, pos_weight, config):
    results = []
    best_overall_acc = 0
    best_params = None

    print(f"{'Activation':<10} | {'Opt':<8} | {'LR':<7} | {'Epoch':<5} | {'Val Loss':<8} | {'B. Acc'}")
    print("-" * 75)

    for act_name, act_class in config['activations'].items():
        for opt_name, opt_class in config['optimizers'].items():
            for lr in config['learning_rates']:
                
                # Inizializzazione
                model = QuantumMLP(input_dim=input_dim, activation_fn=act_class)
                optimizer = opt_class(model.parameters(), lr=lr, weight_decay=1e-4)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                # Training
                best_state, b_loss, last_epoch = train_model(
                    model, train_loader, X_test_t, y_test_t, criterion, optimizer
                )
                
                # Valutazione finale
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_t).squeeze()
                    y_pred = (logits > 0).float().numpy()
                    acc = balanced_accuracy_score(y_test_raw, y_pred)
                
                results.append((act_name, opt_name, lr, acc))
                print(f"{act_name:<10} | {opt_name:<8} | {lr:<7} | {last_epoch:<5} | {b_loss:<8.4f} | {acc:.2%}")

                if acc > best_overall_acc:
                    best_overall_acc = acc
                    best_params = (act_name, opt_name, lr)
                    
    return best_params, best_overall_acc

def train_final_model(best_params, search_config, input_dim, train_loader, X_test_t, y_test_t, pos_weight, epochs=450):
    # Estrazione parametri dai dizionari di configurazione
    act_name, opt_name, lr = best_params
    act_class = search_config['activations'][act_name]
    opt_class = search_config['optimizers'][opt_name]
    
    # Inizializzazione
    model = QuantumMLP(input_dim=input_dim, activation_fn=act_class)
    optimizer = opt_class(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {'train_loss': [], 'test_loss': [], 'variance': []}
    
    print(f"Avvio training finale: {act_name} + {opt_name} per {epochs} epoche...")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y.view(-1))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validazione
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t).view(-1)
            avg_test_loss = criterion(val_outputs, y_test_t.view(-1)).item()
        
        # Salvataggio metriche
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['variance'].append(abs(avg_test_loss - avg_train_loss))
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
            
    return model, history

def plot_convergence_analysis(history, best_params):
    act_name, opt_name, _ = best_params
    
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Bias (Train Loss)', color='royalblue', lw=2)
    plt.plot(history['test_loss'], label='Errore Totale (Test Loss)', color='forestgreen', lw=2)
    plt.plot(history['variance'], label='Varianza (Test-Train Gap)', color='tomato', linestyle='--', alpha=0.7)

    #plt.yscale('log')
    plt.title(f"Analisi di Convergenza: {act_name} + {opt_name}", fontsize=14)
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Loss (BCE With Logits)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()
    
def esegui_mlp_e_plotta(model, X_test_t, y_test_raw, nome_target, mappa_colore, ax_matrix, ax_text):

    model.eval()
    with torch.no_grad():
        # Calcolo predizioni (Logits > 0 -> Classe 1)
        logits = model(X_test_t).squeeze()
        y_pred = (logits > 0).int().numpy() 
    
    # Etichette coerenti con lo studio fisico
    labels = ['Separabile/Locale', 'Entangled/Viola Bell']

    cm = confusion_matrix(y_test_raw, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=mappa_colore, cbar=False, ax=ax_matrix,
                xticklabels=labels, yticklabels=labels)
    
    acc = accuracy_score(y_test_raw, y_pred)
    ax_matrix.set_title(f'MLP: {nome_target}\nAcc: {acc:.2%}', fontsize=14, fontweight='bold')
    ax_matrix.set_xlabel('Predetto')
    ax_matrix.set_ylabel('Reale')

    report = classification_report(y_test_raw, y_pred, target_names=labels, zero_division=0)
    ax_text.text(0.01, 0.5, report, {'fontsize': 11, 'fontfamily': 'monospace'}, va='center')
    ax_text.axis('off')