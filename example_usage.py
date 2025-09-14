"""
Exemplo de uso do classificador de tumores cerebrais
"""

from brain_tumor_mlp import BrainTumorMLPClassifier
import os

def example_prediction():
    """
    Exemplo de como usar o modelo para predizer uma imagem
    """
    # Inicializar classificador
    classifier = BrainTumorMLPClassifier()
    
    # Carregar modelo pré-treinado
    try:
        classifier.load_model('brain_tumor_mlp_model.pkl')
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print("Modelo não encontrado. Execute primeiro o script principal.")
        return
    
    # Exemplo de predição com uma imagem de teste
    test_image_path = "Testing/glioma/Te-gl_0001.jpg"  # Ajuste o caminho conforme necessário
    
    if os.path.exists(test_image_path):
        try:
            predicted_class, probabilities = classifier.predict_single_image(test_image_path)
            
            print(f"\nImagem: {test_image_path}")
            print(f"Classe predita: {predicted_class}")
            print("\nProbabilidades por classe:")
            print(probabilities)
            
        except Exception as e:
            print(f"Erro na predição: {e}")
    else:
        print(f"Imagem não encontrada: {test_image_path}")

def batch_prediction_example():
    """
    Exemplo de predição em lote
    """
    classifier = BrainTumorMLPClassifier()
    
    try:
        classifier.load_model('brain_tumor_mlp_model.pkl')
    except FileNotFoundError:
        print("Modelo não encontrado. Execute primeiro o script principal.")
        return
    
    # Diretório com imagens para teste
    test_dir = "Testing/glioma"  # Exemplo com glioma
    
    if os.path.exists(test_dir):
        results = []
        
        # Processar algumas imagens
        for i, filename in enumerate(os.listdir(test_dir)[:5]):  # Apenas 5 imagens
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, filename)
                
                try:
                    predicted_class, probabilities = classifier.predict_single_image(image_path)
                    results.append({
                        'filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': probabilities.iloc[0]['Probabilidade']
                    })
                except Exception as e:
                    print(f"Erro ao processar {filename}: {e}")
        
        # Mostrar resultados
        print("\nResultados da predição em lote:")
        print("-" * 50)
        for result in results:
            print(f"Arquivo: {result['filename']}")
            print(f"Predição: {result['predicted_class']}")
            print(f"Confiança: {result['confidence']:.4f}")
            print("-" * 30)
    
    else:
        print(f"Diretório não encontrado: {test_dir}")

if __name__ == "__main__":
    print("EXEMPLO DE USO DO CLASSIFICADOR DE TUMORES CEREBRAIS")
    print("=" * 55)
    
    print("\n1. Predição de imagem única:")
    example_prediction()
    
    print("\n2. Predição em lote:")
    batch_prediction_example()