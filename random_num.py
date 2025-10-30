import random
# print(help(random)) # Display the help documentation for the random module
low=1
high=100
options=['Rock', 'Paper', 'Scissors']
cards=['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King'] 
random.shuffle(cards) # Shuffle the list of cards 
print (cards)
random.choice(options)  # Randomly select an option from the list 
number=random.randint(low,high)  # Generate a random integer between 1 and 100
random.random()  # Generate a random float between 0.0 and 1.0 
print(number)  