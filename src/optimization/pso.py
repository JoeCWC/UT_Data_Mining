import numpy as np

'''
粒子群演算法 (Particle Swarm Optimization, PSO)
📌 用途
這段程式碼實作了一個 粒子群演算法 (PSO)，主要用於 超參數最佳化 或 函數最佳化。
在機器學習中，常用來搜尋模型的最佳參數組合，例如 XGBoost 的 、、 等。

🎯 目的
• 	自動化搜尋最佳解：避免人工調參的低效率。
• 	全域最佳化能力：比起單純的隨機搜尋或網格搜尋，PSO 更容易跳出局部最佳解。
• 	適用於黑箱函數：只需要能計算「適應度 (fitness)」，不需要知道函數的數學形式或梯度。

⚙️ 原理
PSO 的靈感來自 鳥群覓食行為。
每個「粒子」代表一個候選解，粒子會根據以下規則更新位置：
1. 	初始化
• 	在參數空間內隨機生成多個粒子的位置  和速度 。
• 	每個粒子都有自己的最佳解 ，整個群體也有一個全域最佳解 。
2. 	速度與位置更新
每一代迭代時，粒子會根據三個因素更新速度與位置：
• 	慣性項 (w)：保持原本的移動方向。
• 	認知項 (c1)：靠近自己歷史最佳解 。
• 	社會項 (c2)：靠近群體最佳解 。
2. 	更新公式：
v_{i}(t+1) = w \cdot v_{i}(t) + c_1 \cdot r_1 \cdot (pbest_i - x_i) + c_2 \cdot r_2 \cdot (gbest - x_i)
x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)
2. 	其中 r_1, r_2 為隨機數，增加隨機性。
3. 	適應度評估 (fitness function)
• 	每個粒子的新位置會帶入  計算分數（例如 AUC）。
• 	若分數比歷史最佳還好，就更新 ；若比全域最佳還好，就更新 。
4. 	收斂
• 	經過多次迭代後，粒子群會逐漸收斂到一個高分區域，得到近似最佳解。

✅ 優點
• 	不需要梯度資訊，適合黑箱函數。
• 	全域搜尋能力強，能避免陷入局部最佳解。
• 	實作簡單，參數少（w, c1, c2）。
⚠️ 缺點
• 	收斂速度可能比貝葉斯最佳化慢。
• 	需要多次評估 fitness，計算成本高。

'''

class PSO:
    def __init__(self, fitness_func, dim, bounds, num_particles=10, max_iter=10,
                 w=0.7, c1=1.5, c2=1.5):
        self.fitness_func = fitness_func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2

        self.X = np.random.uniform(bounds[:,0], bounds[:,1], (num_particles, dim))
        self.V = np.random.uniform(-1, 1, (num_particles, dim))
        self.pbest = self.X.copy()
        self.pbest_scores = np.array([fitness_func(x) for x in self.X])
        self.gbest = self.pbest[np.argmax(self.pbest_scores)]
        self.gbest_score = max(self.pbest_scores)

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.V[i] = (self.w * self.V[i] +
                             self.c1 * r1 * (self.pbest[i] - self.X[i]) +
                             self.c2 * r2 * (self.gbest - self.X[i]))
                self.X[i] = np.clip(self.X[i] + self.V[i], self.bounds[:,0], self.bounds[:,1])

                score = self.fitness_func(self.X[i])
                if score > self.pbest_scores[i]:
                    self.pbest[i], self.pbest_scores[i] = self.X[i], score
                    if score > self.gbest_score:
                        self.gbest, self.gbest_score = self.X[i], score

            print(f"Iter {t+1}/{self.max_iter} | Best AUC={self.gbest_score:.4f}")

        return self.gbest, self.gbest_score