1. Mecanismo de Sincronização
        


2. Exclusão Múeua (Distribuida)
    Será necessário exclusão mútua porque os foldes depois de realizados precisam ser recombinados para avaliação final do modelo, e é o líder que vai responsável por isso. Enatão o modelo centralizado encaixa perfeitamente, pois ele concetrará a operação e o gerenciamento.


3. Será necessário algum algoritimo de Eleição? Qual
    #Sim, como teremos um líder, que planejamos que seja tolerante a cadas, será necessário eleger um novo líder para realizar o trabalho de delegação de folds. Acreditamos que um mecanismo simples será útil.
    #Como a quantidade de máquinas não é muito dinamico, nem tem muitas conexões, a chance de falha é baixa, então um mecanismo desse é mais que suficiente.


4. Se vai usar pubsub, como será implementado
