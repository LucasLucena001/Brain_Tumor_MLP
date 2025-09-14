"""
Análise Avançada para Projeto de Pós-Graduação
Comparações com diferentes abordagens e análises estatísticas profundas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from brain_tumor_mlp import BrainTumorMLPClassifier

class AdvancedBrainTumorAnalysis:
    def __init__(self):
        self.classifier = BrainTumorMLPClassifier(img_size=(64, 64))
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Carrega os dados pré-processados"""
        print("Carregando dados para análise avançada...")
        X_train, X_test, y_train, y_test = self.classifier.load_and_preprocess_data()
        return X_train, X_test, y_train, y_test
    
    def compare_multiple_algorithms(self, X_train, X_test, y_train, y_test):
        """
        Compara MLP com outros algoritmos de machine learning
        """
        print("\n=== COMPARAÇÃO DE ALGORITMOS ===")
        
        # Definir modelos para comparação
        models = {
            'MLP': self.classifier.mlp if self.classifier.mlp else 
                   self.classifier.train_mlp(optimize_hyperparameters=False),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results_comparison = []
        
        for name, model in models.items():
            print(f"Treinando {name}...")
            
            if name != 'MLP':  # MLP já foi treinado
                model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results_comparison.append({
                'Algoritmo': name,
                'Acurácia': accuracy
            })
            
            self.models[name] = model
            
            print(f"{name} - Acurácia: {accuracy:.4f}")
        
        # Visualizar comparação
        df_comparison = pd.DataFrame(results_comparison)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_comparison['Algoritmo'], df_comparison['Acurácia'])
        plt.title('Comparação de Algoritmos de Classificação')
        plt.ylabel('Acurácia')
        plt.ylim(0, 1)
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, df_comparison['Acurácia']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_comparison
    
    def learning_curve_analysis(self, X_train, y_train):
        """
        Análise de curvas de aprendizado
        """
        print("\n=== ANÁLISE DE CURVAS DE APRENDIZADO ===")
        
        # Curva de aprendizado para MLP
        train_sizes, train_scores, val_scores = learning_curve(
            self.classifier.mlp, X_train, y_train, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Calcular médias e desvios padrão
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plotar curva de aprendizado
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Treinamento')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validação')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Tamanho do Conjunto de Treinamento')
        plt.ylabel('Acurácia')
        plt.title('Curva de Aprendizado - MLP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return train_sizes, train_scores, val_scores
    
    def validation_curve_analysis(self, X_train, y_train):
        """
        Análise de curvas de validação para diferentes hiperparâmetros
        """
        print("\n=== ANÁLISE DE CURVAS DE VALIDAÇÃO ===")
        
        # Curva de validação para alpha (regularização)
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0]
        
        train_scores, val_scores = validation_curve(
            self.classifier.mlp, X_train, y_train,
            param_name='alpha', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Treinamento')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validação')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Parâmetro Alpha (Regularização)')
        plt.ylabel('Acurácia')
        plt.title('Curva de Validação - Parâmetro Alpha')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('validation_curve_alpha.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return param_range, train_scores, val_scores
    
    def dimensionality_reduction_analysis(self, X_train, y_train):
        """
        Análise com redução de dimensionalidade
        """
        print("\n=== ANÁLISE DE REDUÇÃO DE DIMENSIONALIDADE ===")
        
        # PCA
        print("Aplicando PCA...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        
        # t-SNE
        print("Aplicando t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_train[:1000])  # Usar subset para t-SNE
        y_tsne = y_train[:1000]
        
        # Visualizar
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
        axes[0].set_title('PCA - Redução para 2D')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} da variância)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} da variância)')
        
        # t-SNE
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='viridis', alpha=0.6)
        axes[1].set_title('t-SNE - Redução para 2D')
        axes[1].set_xlabel('Dimensão 1')
        axes[1].set_ylabel('Dimensão 2')
        
        # Colorbar
        plt.colorbar(scatter, ax=axes[0])
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise da variância explicada pelo PCA
        pca_full = PCA()
        pca_full.fit(X_train)
        
        plt.figure(figsize=(10, 6))
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Acumulada')
        plt.title('Variância Explicada pelo PCA')
        plt.grid(True, alpha=0.3)
        
        # Linha para 95% da variância
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% da variância')
        plt.legend()
        
        plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca, tsne, X_pca, X_tsne
    
    def statistical_analysis(self, results_comparison):
        """
        Análise estatística dos resultados
        """
        print("\n=== ANÁLISE ESTATÍSTICA ===")
        
        # Teste de normalidade dos resultados
        accuracies = results_comparison['Acurácia'].values
        
        # Teste de Shapiro-Wilk para normalidade
        shapiro_stat, shapiro_p = stats.shapiro(accuracies)
        print(f"Teste de Shapiro-Wilk:")
        print(f"  Estatística: {shapiro_stat:.4f}")
        print(f"  p-valor: {shapiro_p:.4f}")
        
        if shapiro_p > 0.05:
            print("  Os dados seguem distribuição normal (p > 0.05)")
        else:
            print("  Os dados não seguem distribuição normal (p <= 0.05)")
        
        # Estatísticas descritivas
        print(f"\nEstatísticas Descritivas das Acurácias:")
        print(f"  Média: {np.mean(accuracies):.4f}")
        print(f"  Mediana: {np.median(accuracies):.4f}")
        print(f"  Desvio Padrão: {np.std(accuracies):.4f}")
        print(f"  Variância: {np.var(accuracies):.4f}")
        print(f"  Mínimo: {np.min(accuracies):.4f}")
        print(f"  Máximo: {np.max(accuracies):.4f}")
        
        # Intervalo de confiança para a média
        confidence_interval = stats.t.interval(
            0.95, len(accuracies)-1, 
            loc=np.mean(accuracies), 
            scale=stats.sem(accuracies)
        )
        
        print(f"  Intervalo de Confiança (95%): [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'confidence_interval': confidence_interval
        }
    
    def generate_advanced_report(self, all_results):
        """
        Gera relatório avançado para pós-graduação
        """
        print("\n" + "="*80)
        print("RELATÓRIO AVANÇADO - ANÁLISE COMPARATIVA DE ALGORITMOS")
        print("PARA CLASSIFICAÇÃO DE TUMORES CEREBRAIS")
        print("="*80)
        
        print("\n1. METODOLOGIA:")
        print("   - Dataset: Brain MRI Images for Brain Tumor Detection")
        print("   - Classes: glioma, meningioma, notumor, pituitary")
        print("   - Pré-processamento: Redimensionamento, normalização, padronização")
        print("   - Validação: Validação cruzada 5-fold")
        
        print("\n2. ALGORITMOS COMPARADOS:")
        for result in all_results['comparison']:
            print(f"   - {result['Algoritmo']}: {result['Acurácia']:.4f}")
        
        print("\n3. ANÁLISE ESTATÍSTICA:")
        stats_results = all_results['statistical']
        print(f"   - Média das acurácias: {stats_results['mean']:.4f}")
        print(f"   - Desvio padrão: {stats_results['std']:.4f}")
        print(f"   - Intervalo de confiança (95%): [{stats_results['confidence_interval'][0]:.4f}, {stats_results['confidence_interval'][1]:.4f}]")
        
        print("\n4. CONCLUSÕES E RECOMENDAÇÕES:")
        best_algorithm = max(all_results['comparison'], key=lambda x: x['Acurácia'])
        print(f"   - Melhor algoritmo: {best_algorithm['Algoritmo']} ({best_algorithm['Acurácia']:.4f})")
        
        if best_algorithm['Algoritmo'] == 'MLP':
            print("   - O MLP demonstrou ser eficaz para esta tarefa")
            print("   - Recomenda-se otimização adicional de hiperparâmetros")
        else:
            print(f"   - {best_algorithm['Algoritmo']} superou o MLP")
            print("   - Considerar ensemble methods para melhor performance")
        
        print("\n5. TRABALHOS FUTUROS:")
        print("   - Implementar técnicas de data augmentation")
        print("   - Testar arquiteturas de deep learning (CNN)")
        print("   - Explorar transfer learning com modelos pré-treinados")
        print("   - Implementar explicabilidade do modelo (LIME, SHAP)")
        
        print("\n" + "="*80)


def main():
    """
    Executa análise avançada completa
    """
    print("ANÁLISE AVANÇADA - CLASSIFICAÇÃO DE TUMORES CEREBRAIS")
    print("="*60)
    
    # Inicializar análise
    analysis = AdvancedBrainTumorAnalysis()
    
    # Carregar dados
    X_train, X_test, y_train, y_test = analysis.load_data()
    
    # Treinar MLP base
    analysis.classifier.train_mlp(optimize_hyperparameters=False)
    
    # 1. Comparar algoritmos
    comparison_results = analysis.compare_multiple_algorithms(X_train, X_test, y_train, y_test)
    
    # 2. Análise de curvas de aprendizado
    learning_results = analysis.learning_curve_analysis(X_train, y_train)
    
    # 3. Análise de curvas de validação
    validation_results = analysis.validation_curve_analysis(X_train, y_train)
    
    # 4. Análise de redução de dimensionalidade
    dim_reduction_results = analysis.dimensionality_reduction_analysis(X_train, y_train)
    
    # 5. Análise estatística
    statistical_results = analysis.statistical_analysis(comparison_results)
    
    # 6. Relatório final
    all_results = {
        'comparison': comparison_results.to_dict('records'),
        'statistical': statistical_results
    }
    
    analysis.generate_advanced_report(all_results)
    
    print("\nArquivos gerados:")
    print("- algorithm_comparison.png")
    print("- learning_curve.png")
    print("- validation_curve_alpha.png")
    print("- dimensionality_reduction.png")
    print("- pca_variance_explained.png")


if __name__ == "__main__":
    main()