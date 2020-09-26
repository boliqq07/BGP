# x = np.linspace(0, 1, 1000)
# y = np.sin(10 * np.pi * x) + x
# mine = MINE(alpha=0.6, c=15)
# mine.compute_score(x, y)
#
# print("Without noise:")
# print("MIC", mine.mic())
# print()
#
# np.random.seed(0)
# y += np.random.uniform(-1, 1, x.shape[0])  # add some noise
# mine.compute_score(x, y)
#
#
# def MIC_matirx(dataframe, mine):
#
#     data = np.array(dataframe)
#     n = len(data[0, :])
#     result = np.zeros([n, n])
#
#     for i in range(n):
#         for j in range(n):
#             mine.compute_score(data[:, i], data[:, j])
#             result[i, j] = mine.mic()
#             result[j, i] = mine.mic()
#     RT = result
#     return RT
#
#
# mine = MINE(alpha=0.6, c=5)
# data_wine_mic = MIC_matirx(
#     np.array([[1, 2, 3, 4, 5, 1, 5, 2, 1, 4], [2, 3, 4, 1, 2, 1, 3, 1, 6, 1], [6, 4, 5, 3, 4, 1, 4, 1, 6, 2]]).T,
#     mine)
u = []
for i in range(10):
    u += i
