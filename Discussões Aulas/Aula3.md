***

## **1. Nomeação (Naming)**

Para garantir que o sistema suporte a entrada/saída dinâmica de nós e a recuperação de falhas do Líder, adotaremos um modelo que mistura a identificação nativa do middleware com um gerenciamento de estado compartilhado.

- **Quais recursos serão nomeados?**
    - **Processos (Nós):** Cada instância na rede (Líder ou Seguidor).
    - **Unidades de Trabalho (Folds):** Cada partição do K-Fold Cross-Validation.
        
- **Esquema de Nomeação:** **Nomeação Plana (Flat Naming)**.
    - **Nós:** Identificados pelo `rank` único (inteiro de $0$ a $n-1$) fornecido pelo `MPI_COMM_WORLD` do OpenMPI.
    - **Folds:** Identificados por IDs numéricos simples (Ex: Fold 1, Fold 2), o que simplifica o rastreio na fila de tarefas.
        
- **Mecanismo de Resolução de Nomes:** **Baseado em Tabela/Índice Globalmente Replicado**.
    - Diferente de um sistema mestre-escravo tradicional, todos os nós mantêm uma tabela em memória que mapeia o `ID_do_Fold` $\rightarrow$ `Estado` (Pendente, Em Execução, Concluído) $\rightarrow$ `Rank_do_Processo`.
    - **Sincronização:** Quando um Seguidor conclui um treinamento, ele realiza um **broadcast** do resultado para todos os outros nós. Isso garante que, caso o Líder atual falhe, qualquer outro nó possua as informações necessárias para assumir a coordenação com a tabela de nomes atualizada.

---

## **2. Processos**

### **Uso de Threads**
- **Nos Seguidores:** **Não** se recomenda a criação de threads manuais no nível da aplicação para o treinamento concorrente de múltiplos folds. Como bibliotecas de IA (PyTorch/TensorFlow) já utilizam threads internas otimizadas (BLAS/MKL), cada nó deve processar **um único fold por vez** para evitar a contenção de CPU e degradação de performance.
- **Nos Seguidores:** **Não** se recomenda a criação de threads manuais no nível da aplicação para o treinamento concorrente de múltiplos folds. Como bibliotecas de IA (PyTorch/TensorFlow) já utilizam threads internas otimizadas (BLAS/MKL), cada nó deve processar **um único fold por vez** para evitar a contenção de CPU e degradação de performance.

- **No Líder:** O uso de threads é viável para gerenciar o **Watchdog (Monitor de Timeout)** e a escuta de mensagens de rede sem bloquear a lógica principal de controle.
### **Servidores Stateful vs. Stateless**
- **Modelo: Stateful**.
    - Devido à necessidade de tolerância a falhas do Líder, todos os nós devem ser **Stateful**.
    - Eles armazenam o estado atual do treinamento (quais folds faltam e quem os está processando) para garantir a **consistência eventual** e permitir que a liderança seja transferida sem perda de progresso.
### **Virtualização e Ambientes**
- **Uso de Containers (Docker):** É essencial para encapsular o ecossistema complexo de Python, OpenMPI e drivers de IA (como CUDA).
- **Justificativa:** Garante a **idempotência** do ambiente de execução; um fold treinado no Nó A terá o mesmo comportamento técnico se for reatribuído ao Nó B após um timeout, eliminando erros por divergência de versões de bibliotecas.