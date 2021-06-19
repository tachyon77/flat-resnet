def generate_data():
    for i in range(5):        
        yield i+1


x = generate_data()

for a in x:
    print(a)

