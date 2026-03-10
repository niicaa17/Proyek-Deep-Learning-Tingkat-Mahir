def kalkulasi_rumit_eager(x):
  total = tf.constant(0.0)
  for i in tf.range(100000):
    total += tf.cast(i, dtype=tf.float32) * x
  return total
 
 
start_time = time.time()
hasil = kalkulasi_rumit_eager(tf.constant(2.0))
end_time = time.time()
 
 
print(f"Hasil Eager Mode: {hasil.numpy()}")
print(f"Waktu Eksekusi Eager Mode: {end_time - start_time:.4f} detik")

@tf.function
def kalkulasi_rumit_graph(x):
  total = tf.constant(0.0)
  for i in tf.range(100000):
    total += tf.cast(i, dtype=tf.float32) * x
  return total
 
# Mengukur waktu eksekusi pertama kali
print("Eksekusi Pertama Graph (Kompilasi)")
start_time = time.time()
hasil = kalkulasi_rumit_graph(tf.constant(2.0))
end_time = time.time()
 
print(f"Hasil Graph Mode: {hasil.numpy()}")
print(f"Waktu Eksekusi Pertama Graph: {end_time - start_time:.4f} detik")
 
# Mengukur waktu eksekusi kedua
print("Eksekusi Kedua Graph")
start_time = time.time()
hasil = kalkulasi_rumit_graph(tf.constant(2.0))
end_time = time.time()
 
print(f"Hasil Graph Mode: {hasil.numpy()}")
print(f"Waktu Eksekusi Kedua Graph: {end_time - start_time:.4f} detik")

def penjumlahan(a, b):
    return a + b
 
x = tf.constant(2)
y = tf.constant(5)
 
print(penjumlahan(x, y))