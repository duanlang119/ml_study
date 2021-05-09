class Bayes(object):
    def __init__(self):
        self._container = dict()
    def Set(self,hypothis,prob):
        self._container[hypothis]=prob
    def Mult(self,hypothis,prob):
        old_prob = self._container[hypothis]
        self._container[hypothis] = old_prob*prob
    def Normalize(self):
        count = 0
        for hypothis in self._container.values():
            count=count+hypothis
        for hypothis,prob in self._container.items():
            self._container[hypothis]=self._container[hypothis]/count
    def Prob(self,hypothis):
        Prob = self._container[hypothis]
        return Prob
#实例化Bayes类
bayes = Bayes()

#先验概率
bayes.Set('Bow_A',0.5) #P(碗A)=1/2
bayes.Set('Bow_B',0.5) #P(碗B)=1/2

#后验概率
bayes.Mult('Bow_A',0.75) #P(香草饼|碗A)=3/4
bayes.Mult('Bow_B',0.5) #P(香草饼|碗B)=1/2

bayes.Normalize()
prob = bayes.Prob('Bow_A')#P(碗A|香草饼)
print('从碗A取到香草曲奇饼的概率:{}'.format(prob))
