from Models.plant import Plant


baseFile = open('./base/data_Mar_64.txt', 'r')
columns = 64

plants = {}

minValue = [1]*columns
maxValue = [-1]*columns

for row in baseFile.readlines():
    data = row.strip().split(',')
    plant = Plant(data[0], [float(i) for i in data[1::]])

    for valueIndex in range(len(plant.vectors)):
        if plant.vectors[valueIndex] > maxValue[valueIndex]:
            maxValue[valueIndex] = plant.vectors[valueIndex]
        elif plant.vectors[valueIndex] < minValue[valueIndex]:
            minValue[valueIndex] = plant.vectors[valueIndex]

    if plant.name not in plants:
        plants[plant.name] = []

    plants[plant.name].append(plant)

baseFile.close()
normalizedBase = open('./base/normalized_data.txt', 'w')

min = -1
max = 1

for name in plants:
    for plant in plants[name]:
        normalizedBase.write(plant.name)
        for vectorIndex in range(len(plant.vectors)):
            plant.vectors[vectorIndex] = ((plant.vectors[vectorIndex] - minValue[vectorIndex]) / (maxValue[vectorIndex] - minValue[vectorIndex])) * (max - min) + min
            normalizedBase.write(',' + str(plant.vectors[vectorIndex]))
        normalizedBase.write('\n')

normalizedBase.close()
