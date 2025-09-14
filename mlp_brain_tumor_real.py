"""
Gerador de Gráficos Individuais para Relatório
Classificação de Tumores Cerebrais usando MLP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from brain_tumor_mlp import BrainTumorMLPClassifier

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

class GeradorGraficosIndividuais:
    def __init__(self):
        self.classifier = BrainTumorMLPClassifier(img_size=(64, 64))
        self.resultados = {}
        
    def executar_analise_completa(self):
        """Executa análise completa e coleta resultados"""
        print("🔄 Executando análise completa...")
        
        # Carregar e analisar dados
        X_train, X_test, y_train, y_test = self.classifier.load_and_preprocess_data()
        dataset_info = self.classifier.analyze_dataset()
        
        # Treinar modelo
        self.classifier.train_mlp(optimize_hyperparameters=False)
        
        # Avaliar modelo
        results = self.classifier.evaluate_model()
        
        # Validação cruzada
        cv_results = self.classifier.cross_validation_analysis()
        
        # Análise de features
        importance_map = self.classifier.feature_importance_analysis()
        
        self.resultados = {
            'dataset_info': dataset_info,
            'model_results': results,
            'cv_results': cv_results,
            'importance_map': importance_map
        }
        
        return self.resultados
    
    def grafico_1_distribuicao_dataset(self):
        """Gráfico 1: Distribuição do Dataset"""
        print("📊 Gerando Gráfico 1: Distribuição do Dataset...")
        
        dataset_info = self.resultados['dataset_info']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        classes = dataset_info['Classe']
        treino = dataset_info['Treinamento']
        teste = dataset_info['Teste']
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, treino, width, label='Treinamento', 
                       color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, teste, width, label='Teste', 
                       color='lightcoral', alpha=0.8)
        
        ax1.set_title('Distribuição das Classes por Conjunto', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Número de Imagens')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de pizza
        total = dataset_info['Total']
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        wedges, texts, autotexts = ax2.pie(total, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Distribuição Total das Classes', fontsize=16, fontweight='bold')
        
        # Melhorar aparência do texto
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('grafico_01_distribuicao_dataset.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return dataset_info
    
    def grafico_2_performance_modelo(self):
        """Gráfico 2: Performance do Modelo"""
        print("📈 Gerando Gráfico 2: Performance do Modelo...")
        
        results = self.resultados['model_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Acurácias
        metricas = ['Acurácia\nTreinamento', 'Acurácia\nTeste']
        valores = [results['train_accuracy'], results['test_accuracy']]
        cores = ['lightgreen', 'orange']
        
        bars = ax1.bar(metricas, valores, color=cores, alpha=0.8, width=0.6)
        ax1.set_title('Acurácia do Modelo', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Acurácia')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, valor in zip(bars, valores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{valor:.3f}\n({valor*100:.1f}%)', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        # Métricas por classe
        metrics_df = results['metrics_df']
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax2.bar(x - width, metrics_df['Precisão'], width, label='Precisão', alpha=0.8)
        ax2.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax2.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax2.set_title('Métricas por Classe', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df['Classe'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('grafico_02_performance_modelo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def grafico_3_matriz_confusao(self):
        """Gráfico 3: Matriz de Confusão"""
        print("🎯 Gerando Gráfico 3: Matriz de Confusão...")
        
        results = self.resultados['model_results']
        cm = results['confusion_matrix']
        classes = self.classifier.label_encoder.classes_
        
        plt.figure(figsize=(10, 8))
        
        # Criar heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Número de Predições'})
        
        plt.title('Matriz de Confusão - Classificação de Tumores Cerebrais', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Classe Predita', fontsize=14)
        plt.ylabel('Classe Real', fontsize=14)
        
        # Adicionar estatísticas
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Acurácia Geral: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig('grafico_03_matriz_confusao.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def grafico_4_validacao_cruzada(self):
        """Gráfico 4: Resultados da Validação Cruzada"""
        print("🔄 Gerando Gráfico 4: Validação Cruzada...")
        
        cv_results = self.resultados['cv_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot
        cv_data = [cv_results['Acurácia'], cv_results['Precisão'], 
                  cv_results['Recall'], cv_results['F1-Score']]
        labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        
        bp = ax1.boxplot(cv_data, labels=labels, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax1.set_title('Distribuição das Métricas - Validação Cruzada 5-Fold', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Gráfico de linha com médias
        means = [np.mean(data) for data in cv_data]
        stds = [np.std(data) for data in cv_data]
        
        ax2.errorbar(range(len(labels)), means, yerr=stds, 
                    marker='o', markersize=8, capsize=5, capthick=2, linewidth=2)
        ax2.set_title('Médias e Desvios Padrão - Validação Cruzada', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Adicionar valores
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('grafico_04_validacao_cruzada.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_results
    
    def grafico_5_arquitetura_mlp(self):
        """Gráfico 5: Arquitetura do MLP"""
        print("🧠 Gerando Gráfico 5: Arquitetura do MLP...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Definir arquitetura
        layers = [12288, 200, 100, 50, 4]
        layer_names = ['INPUT\n64×64×3\n(12,288 neurônios)', 'HIDDEN 1\n(200 neurônios)', 
                      'HIDDEN 2\n(100 neurônios)', 'HIDDEN 3\n(50 neurônios)', 
                      'OUTPUT\n(4 classes)']
        colors = ['#FFE4B5', '#E6F3FF', '#E6F3FF', '#E6F3FF', '#E6FFE6']
        
        # Posições
        x_positions = np.linspace(1, 13, len(layers))
        y_center = 4
        
        # Desenhar camadas
        for i, (x, neurons, name, color) in enumerate(zip(x_positions, layers, layer_names, colors)):
            # Retângulo da camada
            width = 1.8
            height = 2
            rect = plt.Rectangle((x-width/2, y_center-height/2), width, height, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Texto da camada
            ax.text(x, y_center, name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', wrap=True)
            
            # Setas entre camadas
            if i < len(layers) - 1:
                ax.arrow(x + width/2, y_center, x_positions[i+1] - x - width, 0, 
                        head_width=0.2, head_length=0.3, fc='gray', ec='gray', linewidth=2)
        
        # Configurações técnicas
        config_text = """CONFIGURAÇÕES TÉCNICAS:
• Função de Ativação: ReLU
• Regularização: L2 (α = 0.001)
• Otimizador: Adam
• Learning Rate: Adaptativo
• Early Stopping: Habilitado
• Max Iterações: 1000
• Validação: 10% dos dados"""
        
        ax.text(1, 1.5, config_text, fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 6)
        ax.set_title('Arquitetura do Multi-Layer Perceptron (MLP)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('grafico_05_arquitetura_mlp.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_6_comparacao_algoritmos(self):
        """Gráfico 6: Comparação com Outros Algoritmos"""
        print("⚖️ Gerando Gráfico 6: Comparação de Algoritmos...")
        
        # Dados simulados (você pode executar advanced_analysis.py para dados reais)
        results = self.resultados['model_results']
        mlp_accuracy = results['test_accuracy']
        
        algoritmos = ['MLP\n(Nosso Modelo)', 'Random Forest', 'SVM\n(RBF)', 'Logistic\nRegression']
        acuracias = [mlp_accuracy, 0.82, 0.78, 0.75]  # Valores estimados para comparação
        cores = ['gold', 'lightblue', 'lightgreen', 'lightcoral']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(algoritmos, acuracias, color=cores, alpha=0.8, width=0.6)
        
        # Destacar nosso modelo
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(3)
        
        ax.set_title('Comparação de Performance entre Algoritmos', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Acurácia', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, acuracias):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{acc:.3f}\n({acc*100:.1f}%)', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12)
        
        # Adicionar linha de referência
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
                  label='Threshold Clínico (80%)')
        ax.legend()
        
        # Nota explicativa
        ax.text(0.02, 0.98, 'Nota: Valores de comparação são estimados para demonstração', 
               transform=ax.transAxes, fontsize=10, style='italic', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('grafico_06_comparacao_algoritmos.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return acuracias
    
    def gerar_todos_graficos(self):
        """Gera todos os gráficos individuais"""
        print("🎨 GERANDO TODOS OS GRÁFICOS INDIVIDUAIS")
        print("="*50)
        
        # Executar análise completa
        self.executar_analise_completa()
        
        # Gerar gráficos
        graficos = [
            self.grafico_1_distribuicao_dataset,
            self.grafico_2_performance_modelo,
            self.grafico_3_matriz_confusao,
            self.grafico_4_validacao_cruzada,
            self.grafico_5_arquitetura_mlp,
            self.grafico_6_comparacao_algoritmos
        ]
        
        resultados_graficos = {}
        
        for i, grafico_func in enumerate(graficos, 1):
            print(f"\n📊 Processando Gráfico {i}/6...")
            resultado = grafico_func()
            resultados_graficos[f'grafico_{i}'] = resultado
        
        print("\n✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
        print("\n📁 Arquivos criados:")
        arquivos_gerados = [
            'grafico_01_distribuicao_dataset.png',
            'grafico_02_performance_modelo.png', 
            'grafico_03_matriz_confusao.png',
            'grafico_04_validacao_cruzada.png',
            'grafico_05_arquitetura_mlp.png',
            'grafico_06_comparacao_algoritmos.png'
        ]
        
        for arquivo in arquivos_gerados:
            if os.path.exists(arquivo):
                print(f"✅ {arquivo}")
            else:
                print(f"❌ {arquivo}")
        
        return resultados_graficos

def main():
    """Função principal"""
    gerador = GeradorGraficosIndividuais()
    resultados = gerador.gerar_todos_graficos()
    
    print(f"\n🎉 PROCESSO CONCLUÍDO!")
    print("Os gráficos estão prontos para uso no seu vídeo e relatório.")

if __name__ == "__main__":
    main()