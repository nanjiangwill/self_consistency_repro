"""
Chain-of-Thought prompts for different datasets.
Based on the examples from the paper.
"""

# GSM8K (Grade School Math) prompt
GSM8K_PROMPT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees in the grove. After the grove workers plant more trees, there will be 21 trees. So the grove workers must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more cars arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74 chocolates. After eating 35, they had 74 - 35 = 39 chocolates left. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. After giving some to Denny, he has 12 lollipops left. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 toys from his mom and 2 toys from his dad. So he got 2 + 2 = 4 toys. Now he has 5 + 4 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were 9 computers in the server room. For each day from monday to thursday, 5 more computers were installed. So 5 * 4 = 20 computers were added. Now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: {question}
A:"""

# SVAMP prompt
SVAMP_PROMPT = """
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia has $23. She bought 5 bagels for $3 each. So she spent 5 * $3 = $15 on bagels. Now she has $23 - $15 = $8 left. The answer is 8.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. After giving some to Denny, he has 12 lollipops left. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 toys from his mom and 2 toys from his dad. So he got 2 + 2 = 4 toys. Now he has 5 + 4 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were 9 computers in the server room. For each day from monday to thursday, 5 more computers were installed. So 5 * 4 = 20 computers were added. Now there are 9 + 20 = 29 computers. The answer is 29.

Q: {question}
A:"""

# AQuA prompt
AQUA_PROMPT = """
Q: The sum of 3 consecutive even numbers is 24. What is the smallest of these numbers?
A: Let's say the three consecutive even numbers are n, n+2, and n+4. We know that n + (n+2) + (n+4) = 24. This simplifies to 3n + 6 = 24. So 3n = 18, which means n = 6. So the three numbers are 6, 8, and 10. The smallest is 6. The answer is 6.

Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?
A: The distance traveled is equal to speed multiplied by time. So the distance is 20 km/hr * 2.5 hr = 50 km. The answer is 50.

Q: How many prime numbers are there between 1 and 10?
A: The prime numbers between 1 and 10 are 2, 3, 5, and 7. There are 4 prime numbers. The answer is 4.

Q: If a fair coin is tossed 4 times, what is the probability of getting exactly 3 heads?
A: When a fair coin is tossed 4 times, there are 2^4 = 16 possible outcomes. To get exactly 3 heads, we need to choose which 3 of the 4 tosses will be heads. This is C(4,3) = 4. So the probability is 4/16 = 1/4. The answer is 1/4.

Q: {question}
A:"""

# StrategyQA prompt
STRATEGYQA_PROMPT = """
Q: Do hamsters provide food for any animals?
A: Hamsters are small rodents that are kept as pets. They are prey animals in the wild. Animals like snakes, birds of prey, and wild cats hunt and eat hamsters. So hamsters do provide food for some animals. The answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields is an actress and model who graduated from Princeton University with a bachelor's degree in French literature. Princeton is an Ivy League school with very high academic standards, similar to the University of Pennsylvania. Since Brooke Shields succeeded at Princeton, she could likely succeed at the University of Pennsylvania as well. The answer is yes.

Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
A: Hydrogen's atomic number is 1. 1 squared is still 1. The Spice Girls were a group of 5 members. 1 does not exceed 5. The answer is no.

Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies typically happen in May or June, which is late spring or early summer in the Northern Hemisphere. Frost is uncommon during these months in most of the United States. However, some colleges in very cold regions or high altitudes might have frost even in late spring. So while it's not common, it can happen at some college commencements. The answer is yes.

Q: {question}
A:"""

# ARC-Challenge prompt
ARC_CHALLENGE_PROMPT = """
Q: Which of these would let the most heat travel through?
(A) a new plastic spoon in a cafeteria
(B) a silver spoon in a cafeteria
(C) a plastic spoon in a cafeteria that was heated in the sun
(D) a silver spoon in a cafeteria that was heated in the sun

A: Heat travels through conductive materials. Silver is a metal and metals are good conductors of heat. Plastic is not a good conductor of heat. Between the silver spoon and plastic spoon, the silver spoon will let more heat travel through. Whether the spoon is heated in the sun doesn't affect its conductivity. So the silver spoon will let the most heat travel through, regardless of whether it was heated. The answer is (B).

Q: Which statement best explains why a green plant wilts when it is not watered?
(A) The concentration of water in the plant is higher than the concentration of water in the air.
(B) The concentration of water in the plant is lower than the concentration of water in the air.
(C) The plant cell membranes cannot get carbon dioxide.
(D) The plant cell membranes cannot make food.

A: Plants need water to maintain their structure and carry out photosynthesis. When a plant is not watered, it loses water through transpiration but doesn't replace it. This causes the plant cells to lose turgor pressure, which makes the plant wilt. The concentration of water in the plant becomes lower than optimal, but this is not about comparison with the air. The plant can still get carbon dioxide through stomata, and while lack of water affects photosynthesis, the immediate cause of wilting is loss of cell turgor. The answer is (A).

Q: A student riding on a train moving at a constant velocity drops a ball. Which of the following best describes the path of the ball as observed by the student?
(A) The ball falls in a straight line.
(B) The ball follows a parabolic path forward.
(C) The ball follows a parabolic path backward.
(D) The ball falls in a straight line and then moves backward.

A: From the perspective of someone on the train (the student), objects inside the train share the same reference frame. When the student drops the ball, both the student and the ball are moving at the same velocity as the train. Since there's no horizontal force acting on the ball relative to the train, the ball will appear to fall straight down from the student's perspective. The answer is (A).

Q: {question}
A:"""

# CommonsenseQA prompt
COMMONSENSEQA_PROMPT = """
Q: What do people use to absorb extra ink from a fountain pen?
(A) shirt pocket
(B) calligrapher's hand
(C) inkwell
(D) desk drawer
(E) blotter

A: Fountain pens can sometimes release too much ink, leading to smudging. To prevent this, people use a blotter, which is an absorbent paper designed to soak up excess ink. A shirt pocket, calligrapher's hand, inkwell, and desk drawer are not designed to absorb ink. The answer is (E).

Q: What home entertainment equipment requires cable?
(A) radio shack
(B) substation
(C) television
(D) cabinet
(E) desk

A: Home entertainment equipment that requires cable would be devices that need a cable connection to receive content. Among the options, only television typically requires a cable connection to receive channels and content. Radio shack is a store, not equipment. A substation is related to electricity distribution. Cabinet and desk are furniture items. The answer is (C).

Q: The fox walked from the city into the forest, what was it looking for?
(A) pretty flowers
(B) hen house
(C) natural habitat
(D) storybook
(E) dense forest

A: Foxes are wild animals that naturally live in forests, which is their natural habitat. When a fox moves from a city (an unnatural environment for it) to a forest, it's likely seeking its natural habitat. While foxes might be interested in hen houses for food, and forests are indeed dense, the most fundamental thing it would be seeking is its natural environment. Pretty flowers and storybooks are not things foxes would seek. The answer is (C).

Q: {question}
A:"""