def get_gpt4_ICE():
    example_0 = """
Hint: Please answer the question requiring inequality as an answer and provide the final inequality, 
e.g., x < 5, -5 < x < -3, -3 < x < -1 or 1 < x < 3 at the end.\n
Question: The solution set of 3x + 4 < 2x - 5 is __\n
Model response: Subtract 2x from both sides: x + 4 > -5. Subtract 4 from both sides: x \u2264 -9.\n
Extracted answer: x < -9
"""
    
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a fraction as an answer and provide the final value, 
e.g., 1/2, 4/5, 7/9 at the end.\n
Question: What percentage of students wear school uniforms?\n
Model response: 5 out of 15 are in school uniform, so 1/3 students wear school uniforms\n
Extracted answer: 1/3
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, 
e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_0, example_1, example_2, example_3, example_4, example_5]

def force_match():
    hint = """Please read the following example.
Here is the "Hint", "Question", "Example answer", and "Model response". Please extract the answer from the "Model response" in the format of the "Example answer".
"""
    
    example_1 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: Given the functions y = -x + 2 and y = kx + 3 intersect at point P, and the y-coordinate of point P is 2.5, the solution for the system of equations with respect to x and y is ______.\n
Example answer: { x = -0.5, y = 2.5 }\n
Model response: Given the functions \\( y = -x + 2 \\) and \\( y = kx + 3 \\), we know that they intersect at point \\( P \\) with a y-coordinate of 2.5. To find the solution for the system of equations with respect to \\( x \\) and \\( y \\), we follow these steps:\n\n1. Set \\( y = 2.5 \\) in both equations since the y-coordinate of point \\( P \\) is 2.5.\n\nFor the first equation:\n\\[ 2.5 = -x + 2 \\]\n\nSolve for \\( x \\):\n\\[ 2.5 = -x + 2 \\]\n\\[ 2.5 - 2 = -x \\]\n\\[ 0.5 = -x \\]\n\\[ x = -0.5 \\]\n\nSo, the coordinates of point \\( P \\) are \\( (-0.5, 2.5) \\).\n\n2. Substitute \\( x = -0.5 \\) and \\( y = 2.5 \\) into the second equation to find \\( k \\):\n\n\\[ 2.5 = k(-0.5) + 3 \\]\n\nSolve for \\( k \\):\n\\[ 2.5 = -0.5k + 3 \\]\n\\[ 2.5 - 3 = -0.5k \\]\n\\[ -0.5 = -0.5k \\]\n\\[ k = 1 \\]\n\nThus, the solution for the system of equations with respect to \\( x \\) and \\( y \\) is \\( (-0.5, 2.5) \\).\n
Extracted answer: { x = -0.5, y = 2.5 }
"""

    example_2 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: The inverse proportional functions \\(y = \\frac{k_1}{x}, y = \\frac{k_2}{x}, y = \\frac{k_3}{x}\\) are shown in the same coordinate system. What is the relationship between the magnitudes of \\(k_1, k_2, k_3\\)?\n
Example answer: \\(k_1 < k_2 < k_3\\)\n
Model response: The relationship between the magnitudes of \\(k_1, k_2, k_3\\) is \\(k_1 > k_2 > k_3\\).\n
Extracted answer: \\(k_1 > k_2 > k_3\\)
"""

    example_3 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: A rectangle has an area of 48 square centimeters. Its length is 4 centimeters more than its width. The rectangle width is _ and the length is _.
Example answer: (3.21cm, 7.21cm)\n
Model response: Let the width be x cm. Then, the length is x + 6 cm. The equation is x(x + 6) = 48, which simplifies to x^2 + 6x - 48 = 0. Solving this quadratic equation using the quadratic formula, we get x = (-6 ± √(36 + 192)) / 2. This gives us x = (-6 ± √228) / 2. Simplifying further, we get x = 2.86 cm. Therefore, the width is 2.86 cm and the length is 2.86 + 6 = 8.86 cm.\n
Extracted answer: (2.86cm, 8.86cm)
"""

    example_4 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Example answer: [2007, 2008]\n
Model response: The line graph saw its maximum peak between 1998 and 1999.\n
Extracted answer: [1998, 1999]
"""

    example_5 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: The solution set of 3x + 4 < 2x - 5 is __\n
Example answer: x \u2264 -9\n
Model response: Subtract 2x from both sides: x + 4 > -5. Subtract 4 from both sides: x < -9.\n
Extracted answer: x \u2264 -9
"""

    example_6 = """
Hint: Please answer the question in the appropriate format as per the prompt.\n
Question: There is a quadratic equation: x^2 - 5x + 6 = 0. x is __\n
Example answer: x_1 = 2, x_2 = 3\n
Model response: Factor the equation: (x - 2)(x - 3) = 0. Set each factor to zero: x - 2 = 0 or x - 3 = 0. Solve for x: x = 2 or x = 3.",
Extracted answer: x_1 = 2, x_2 = 3
"""
    return [hint, example_1, example_2, example_3, example_4, example_5, example_6]