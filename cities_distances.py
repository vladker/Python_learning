# Определяем расстояние между 3 городами с заданными кооридинатами

site = {
    "Moscow": (550, 370),
    "London": (510, 510),
    "Paris": (480, 480)
}
moscow =site['Moscow']
london =site['London']
paris =site['Paris']

moscow_london=(moscow[0]-london[0])**0.5+(moscow[1]-london[1])**0.5
moscow_london.real

distances=dict()
distances['Moscow']={}
distances['Moscow']['London']=moscow_london
print(distances)