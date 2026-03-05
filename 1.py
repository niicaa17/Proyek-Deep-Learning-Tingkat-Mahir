x = 4
x_squared = tf.square(x)
 
print("tanganku {}".format(x_squared))

tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor2 = tf.constant([1, 2, 3])
 
result = tf.add(tensor1, tensor2)
print(result)