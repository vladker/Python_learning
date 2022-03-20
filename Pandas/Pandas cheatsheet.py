import pandas as pd
import numpy as np
anime = pd.read_csv('anime-recommendations-database/anime.csv')
rating = pd.read_csv('anime-recommendations-database/rating.csv')
anime_modified = anime.set_index('name')

# Загрузка CSV-данных
anime = pd.read_csv('anime-recommendations-database/anime.csv')

# Создание датафрейма из данных, введённых вручную
df = pd.DataFrame([[1,'Bob', 'Builder'],
                  [2,'Sally', 'Baker'],
                  [3,'Scott', 'Candle Stick Maker']],
columns=['id','name', 'occupation'])

# Копирование датафрейма
anime_copy = anime.copy(deep=True)

# Сохранить фрейм в csv
# rating[:10].to_csv('saved_ratings.csv', index=False)

# Сохранить фрейм в excel
# anime[:10].to_excel("saved_ratings.xlsx") 


# Я часто вывожу некоторое количество элементов из начала датафрейма где-нибудь в блокноте
print(anime.head(3))
print(rating.tail(1))

# Подсчёт количества строк в датафрейме
print('Подсчёт количества строк в датафрейме', len(anime))

# Подсчёт количества уникальных значений в столбце
print('Подсчёт количества уникальных значений в столбце', len(rating['user_id'].unique()))
# nunique() также позволяет получить уникальные записи
# ratings['user_id'].nunique()

# Получение сведений о датафрейме
anime.info()

# Выводит сведения о типах данных столбцов
# anime.dtypes()

# Вывод статистических сведений о датафрейме
print('Вывод статистических сведений о датафрейме')
print(anime.describe())

# Вывод статистических сведений о датафрейме
print('Вывод статистических сведений о датафрейме')
print(anime.type.value_counts())

# Создание списка или объекта Series на основе значений столбца
print(anime['genre'].tolist())
print(anime['genre'])

# Получение списка значений из индекса
print(anime_modified.index.tolist())

# Получение списка значений столбцов
print(anime.columns.tolist())

# Присоединение к датафрейму нового столбца с заданным значением
anime['train set'] = True

# Создание нового датафрейма из подмножества столбцов
anime[['name','episodes']]

# Удаление заданных столбцов
anime.drop(['anime_id', 'genre', 'members'], axis=1).head()

# Добавление в датафрейм строки с суммой значений из других строк
df = pd.DataFrame([[1,'Bob', 8000],
                  [2,'Sally', 9000],
                  [3,'Scott', 20]], columns=['id','name', 'power level'])
df.append(df.sum(axis=0), ignore_index=True)
# (axis=0) позволяет получать суммы значений из различных строк
# Команда вида df.sum(axis=1) позволяет суммировать значения в столбцах.
# Похожий механизм применим и для расчёта средних значений. Например — df.mean(axis=0).

# Конкатенация двух датафреймов
# разделяем датафрейм на две части
df1 = anime[0:2]
df2 = anime[2:4]
# объединяем эти части
pd.concat([df1, df2], ignore_index=True)

# Слияние датафреймов
# Функция df.merge, которую мы тут рассмотрим, похожа на левое соединение SQL
rating.merge(anime, left_on='anime_id', right_on='anime_id', suffixes=('_left', '_right'))
print(anime)

# Фильтрация
# Получение строк с нужными индексными значениями
print(anime_modified.loc[['Haikyuu!! Second Season','Gintama']])

# Получение строк по числовым индексам
print(anime_modified.iloc[0:3])

# Получение строк по заданным значениям столбцов
print(anime[anime['type'].isin(['TV', 'Movie'])])

# Если нас интересует единственное значение
print(anime[anime['type'] == 'TV'])

# Получение среза датафрейма
anime[1:3]

# Фильтрация по значению
anime[anime['rating'] > 8]

# Сортировка
anime.sort_values('rating', ascending=False)

# Агрегирование
# Функция df.groupby и подсчёт количества записей
print(anime.groupby('type').count())

# Функция df.groupby и агрегирование столбцов различными способами
# Обратите внимание на то, что здесь используется reset_index().
# В противном случае столбец type становится индексным столбцом. В большинстве случаев я рекомендую делать то же самое.
anime.groupby(["type"]).agg({
  "rating": "sum",
  "episodes": "count",
  "name": "last"
}).reset_index()

# Создание сводной таблицы
tmp_df = rating.copy()
tmp_df.sort_values('user_id', ascending=True, inplace=True)
tmp_df = tmp_df[tmp_df.user_id < 10]
tmp_df = tmp_df[tmp_df.anime_id < 30]
tmp_df = tmp_df[tmp_df.rating != -1]
pd.pivot_table(tmp_df, values='rating', index=['user_id'], columns=['anime_id'], aggfunc=np.sum, fill_value=0)

# Очистка данных
# Запись в ячейки, содержащие значение NaN, какого-то другого значения
pivot = pd.pivot_table(tmp_df, values='rating', index=['user_id'], columns=['anime_id'], aggfunc=np.sum)
pivot.fillna(0)

# Отбор случайных образцов из набора данных
anime.sample(frac=0.25)
# Если используется параметр frac=1, то функция позволяет получить аналог исходного датафрейма,
# строки которого будут перемешаны

# Перебор строк датафрейма
for idx,row in anime[:2].iterrows():
    print(idx, row)

# Борьба с ошибкой IOPub data rate exceeded
# Если вы сталкиваетесь с ошибкой IOPub data rate exceeded — попробуйте, при запуске Jupyter Notebook,
# воспользоваться следующей командой:

# jupyter notebook — NotebookApp.iopub_data_rate_limit=1.0e10