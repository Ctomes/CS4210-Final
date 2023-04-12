import os

def read_class(data_file):
	data = []
	with open(data_file) as f:
		current_vehicle = 0
		for i, line in enumerate(f):
			if i == 0:
				# Getting the number of vehicles in the file
				num_vehicles = int(line.split(':')[1].strip())
				data = [{} for j in range(num_vehicles)]
				continue
			
			if line == '\n':
				# New vehicle being made
				current_vehicle += 1
				continue

			key_val = line.split(':')
			key_val = [el.strip() for el in key_val]
			if key_val[0] in ['position_vehicle', 'position_plate'] or 'char' in key_val[0]:
				# Special case for position fields, adds dictionary of values to dictionary
				vals = key_val[1].split()
				data[current_vehicle][key_val[0]] = {}
				data[current_vehicle][key_val[0]]['x'] = int(vals[0])
				data[current_vehicle][key_val[0]]['y'] = int(vals[1])
				data[current_vehicle][key_val[0]]['width'] = int(vals[2])
				data[current_vehicle][key_val[0]]['height'] = int(vals[3])
			else:
				# General case, adds to dictionary
				data[current_vehicle][key_val[0]] = key_val[1]
	return data

data = []
for file in os.listdir('data'):
	if file.endswith('.txt'):
		data.append(read_class('data/' + file))
		#print(file)
		#for vehicle in data:
		#	print(vehicle)
		#print()
print(len(data))