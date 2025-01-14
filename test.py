import os
import spacy
from collections import defaultdict

class MarkovChainPredictor:
    def __init__(self):
        self.model = defaultdict(list)
        self.nlp = spacy.load('ru_core_news_sm')  # Загружаем модель для русского языка

    def load_text(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def build_model(self, text):
        # Лемматизируем текст
        doc = self.nlp(text)
        words = [token.lemma_ for token in doc if not token.is_punct]  # Убираем пунктуацию
        for i in range(len(words) - 1):
            key = tuple(words[max(0, i - 1):i + 1])  # Используем до два предыдущих слова
            self.model[key].append(words[i + 1])

    def predict(self, input_words):
        # Лемматизируем ввод
        doc = self.nlp(input_words)
        words = [token.lemma_ for token in doc if not token.is_punct]

        # Сначала ищем полное совпадение
        for i in range(len(words), 0, -1):
            key = tuple(words[-i:])
            if key in self.model:
                return self.model[key][0]  # Возвращаем первое предсказание

        # Если нет полного совпадения, ищем только по первому слову
        if words[0] in self.model:
            return self.model[words[0]][0]

        return "ничего не найдено"


def main():
    predictor = MarkovChainPredictor()

    # Автоматически ищем файл test.txt в той же директории, где находится скрипт
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'file.txt')

    try:
        text = predictor.load_text(file_path)
        predictor.build_model(text)
        print("Модель успешно построена!")
    except FileNotFoundError as e:
        print(e)
        return

    print("\nВведите слова для предсказания. Напишите 'выход', чтобы завершить.")
    while True:
        input_words = input("\nВаш ввод: ")
        if input_words.lower() == 'выход':
            break
        prediction = predictor.predict(input_words)
        print(f"Ответ программы: {prediction}")

if __name__ == "__main__":
    main()
