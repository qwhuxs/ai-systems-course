"""
Перевірка якості моделі на 10 промптах (лабораторна 4, пункти 1.4 і 1.5).

Надсилає 10 типових для застосунку «Картки з конспекту» запитів локальній моделі
через OpenAI-сумісний API і зберігає відповіді у файл quality_<label>.md.

Запуск (Ollama має працювати):
    pip install openai
    python quality_check.py --model gemma3:4b --label before
    python quality_check.py --model <інша-модель> --label after
"""
import argparse
import json
import time

from openai import OpenAI

PROMPTS = [
    ("Ентропія", "Ентропія джерела за Шенноном — це середня кількість інформації на один символ повідомлення. Обчислюється за формулою H = −Σ p(i)·log2 p(i), вимірюється в бітах. Ентропія максимальна, коли всі символи рівноймовірні: H max = log2 N. Надлишковість R = 1 − H / H max."),
    ("Код Хеммінга", "Код Хеммінга (7,4) кодує 4 інформаційні біти сімома, додаючи 3 перевірні біти. Він виявляє і виправляє одну помилку в кодовому слові. Позиції перевірних бітів — степені двійки: 1, 2, 4. Синдром показує номер позиції з помилкою."),
    ("Wi-Fi стандарти", "Стандарт IEEE 802.11n (Wi-Fi 4) працює в діапазонах 2,4 і 5 ГГц і вперше використав MIMO. 802.11ac (Wi-Fi 5) працює лише в 5 ГГц. 802.11ax (Wi-Fi 6) додав OFDMA, що підвищує ефективність у мережах з великою кількістю пристроїв."),
    ("Laravel MVC", "Laravel — PHP-фреймворк, побудований за шаблоном MVC. Модель (Eloquent) відповідає за роботу з базою даних, представлення (Blade) — за HTML, контролер обробляє запит і з'єднує модель із представленням. Маршрути описують у файлі routes/web.php. Міграції дозволяють змінювати структуру БД кодом."),
    ("UML", "Діаграма варіантів використання показує акторів і функції системи. Діаграма класів описує класи, їхні атрибути, методи та зв'язки: асоціацію, агрегацію, композицію, успадкування. Діаграма послідовності показує обмін повідомленнями між об'єктами в часі."),
    ("SQL JOIN", "INNER JOIN повертає лише рядки, для яких є збіг в обох таблицях. LEFT JOIN повертає всі рядки лівої таблиці, а для відсутніх збігів — NULL. Первинний ключ однозначно ідентифікує рядок, зовнішній ключ посилається на первинний ключ іншої таблиці."),
    ("Трансформери", "Трансформер — архітектура нейромережі, представлена у 2017 році. Ключовий механізм — self-attention, який дозволяє кожному токену враховувати інші токени послідовності. Decoder-only моделі (GPT) використовують причинну маску, щоб не бачити майбутніх токенів."),
    ("Квантизація", "Квантизація зменшує кількість бітів на вагу моделі: з 16 (fp16) до 8, 4 або 2. Це зменшує пам'ять і пришвидшує генерацію на залізі, обмеженому пропускною здатністю пам'яті. Що менше бітів, то більше падає якість; int4 зазвичай — розумний компроміс, а int2 сильно псує модель."),
    ("Історія (не технічна тема)", "Києво-Могилянська академія заснована в 1632 році Петром Могилою об'єднанням Київської братської та Лаврської шкіл. Це був перший вищий навчальний заклад на теренах України. Навчання велося латиною, викладали філософію, богослов'я, риторику."),
    ("Конспект англійською", "TCP is a connection-oriented protocol that guarantees delivery and order of packets using acknowledgements and retransmission. UDP is connectionless and faster, but does not guarantee delivery. TCP uses a three-way handshake: SYN, SYN-ACK, ACK."),
]

TEMPLATE = ('Зроби 5 карток для повторення українською мовою у форматі JSON '
            '{{"cards":[{{"q":"запитання","a":"відповідь"}}]}}. Бери факти лише з конспекту. '
            'Конспект: {}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")  # ключ Ollama не перевіряє
    out_path = f"quality_{args.label}.md"
    lines = [f"# Перевірка якості: {args.model} ({args.label})\n",
             "Параметри: temperature=0.3, max_tokens=1024.\n"]
    total_time, total_tokens, valid_json = 0.0, 0, 0

    for i, (title, notes) in enumerate(PROMPTS, 1):
        prompt = TEMPLATE.format(notes)
        print(f"[{i}/10] {title}…", flush=True)
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:  # сервер не запущено, модель не знайдено тощо
            print("   ПОМИЛКА:", e)
            lines.append(f"## {i}. {title}\n\n**Помилка:** {e}\n")
            continue
        dt = time.perf_counter() - t0
        answer = r.choices[0].message.content or ""
        tokens = r.usage.completion_tokens if r.usage else 0
        total_time += dt
        total_tokens += tokens

        # чи вдалося розібрати JSON з картками
        text = answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            cards = json.loads(text[text.index("{"): text.rindex("}") + 1])["cards"]
            ok = f"так, {len(cards)} карток"
            valid_json += 1
        except Exception:
            ok = "НІ"
        print(f"   {dt:.1f} с, {tokens} токенів, JSON коректний: {ok}")

        lines.append(f"## {i}. {title}\n")
        lines.append(f"**Промпт:** {prompt}\n")
        lines.append(f"**Час:** {dt:.1f} с · **токенів:** {tokens} · **JSON коректний:** {ok}\n")
        lines.append(f"**Відповідь моделі:**\n\n```\n{answer}\n```\n")

    summary = (f"**Разом:** {total_time:.0f} с, {total_tokens} токенів, "
               f"коректний JSON у {valid_json} з 10 відповідей.")
    lines.insert(2, summary + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + summary)
    print("Відповіді збережено у", out_path)


if __name__ == "__main__":
    main()
