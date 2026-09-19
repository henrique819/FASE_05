# Roteiro do vídeo · até 5 minutos

**Formato sugerido:** tela compartilhada com a apresentação (PDF ou PPTX) e câmera da pessoa que apresenta no canto. Nos minutos finais, mostrar o app Streamlit rodando.

**Ritmo:** cerca de 130 palavras por minuto. O texto abaixo tem por volta de 620 palavras (≈ 4min40s), deixando folga para a demonstração do app.

---

### Slide 1 · Capa (0:00 – 0:15)
> Olá! Somos o grupo do Datathon da Fase 5. Analisamos três anos da Pesquisa Extensiva do Desenvolvimento Educacional da Passos Mágicos para responder a uma pergunta: o programa está tirando os alunos do atraso escolar, e como antecipar quem ainda vai precisar de ajuda?

### Slide 2 · Os dados (0:15 – 0:35)
> Trabalhamos com 3.030 avaliações de 2022 a 2024. Padronizamos as três abas, que tinham formatos diferentes, e ligamos o mesmo aluno entre os anos pelo RA. São 1.365 trajetórias, o que nos permitiu olhar não só a foto de cada ano, mas o filme de cada aluno.

### Slide 3 · Defasagem (0:35 – 1:05)
> Começando pela defasagem: em 2022, sete em cada dez alunos estavam atrás da fase ideal para a idade. Em 2024, menos da metade. Os alunos em fase foram de 30 para 54 por cento, e a defasagem severa praticamente desapareceu. Mas, olhando aluno a aluno, 142 alunos que estavam em fase atrasaram no ano seguinte, porque não avançaram justamente no ano em que a idade exigia. Esse é o grupo que o nosso modelo tenta antecipar.

### Slide 4 · Desempenho acadêmico (1:05 – 1:30)
> O desempenho acadêmico não acompanhou. O IDA subiu em 2023 e voltou a cair em 2024, puxado por Português. E o padrão por fase se repete nos três anos: as notas afundam do quinto ao nono ano, nas Fases 2 a 4.

### Slide 5 · Engajamento (1:30 – 2:05)
> O que move o aluno é o engajamento. Quem está no quartil mais engajado tem dois pontos e meio a mais de IDA e um ponto de virada bem maior. O ponto de virada é explicado principalmente pela avaliação psicopedagógica, pelo engajamento e pelo desempenho. E quando o aluno está bem em IDA, IEG, IPS e IPP ao mesmo tempo, o INDE médio chega à faixa Topázio.

### Slide 6 · Sinais de leitura cuidadosa (2:05 – 2:35)
> Três indicadores pedem cuidado. A autoavaliação é alta para quase todos: 47 por cento dos alunos se avaliam bem acima do desempenho real. O indicador psicossocial antecede quedas de engajamento, mas o sinal é fraco. E o IPP confirma a direção da defasagem, mas separa pouco os grupos. Quando o IAN é baixo e o IPP é alto, temos um aluno possivelmente pronto para avançar de fase.

### Slide 7 · Efetividade e permanência (2:35 – 3:05)
> O programa funciona: a proporção de alunos Topázio dobrou em dois anos, e entre os alunos presentes nos três anos a defasagem média caiu de menos 0,83 para menos 0,24. Mas menos da metade dos alunos Quartzo continua no ano seguinte, contra 84 por cento dos Topázio. Quem mais precisa é quem mais sai.

### Slide 8 · Modelo preditivo (3:05 – 3:45)
> Para agir antes, construímos um modelo que usa os indicadores de hoje para prever se o aluno estará defasado, ou piorando, no próximo ciclo. Garantimos que o mesmo aluno nunca estivesse no treino e no teste ao mesmo tempo, e também testamos prevendo 2024 com dados de 2022. O Random Forest teve ROC-AUC de 0,88 e sinaliza 84 por cento dos alunos que de fato entram em risco, contra 65 por cento de acurácia da regra simples. O sinal mais forte é a defasagem que o aluno terá se não avançar de fase.

### Slide 9 + demonstração do app (3:45 – 4:30)
> Publicamos o modelo num app Streamlit.
*(Trocar para o app.)*
> Na primeira aba, a equipe informa os indicadores de um aluno e recebe a probabilidade, a faixa de risco e o motivo em linguagem simples. *(Mudar a fase para 3 e a idade para 15 e mostrar o risco subindo.)* Na segunda, envia a própria planilha do PEDE e recebe o ranking da turma. Na terceira, vê o panorama previsto para 2025.

### Slide 10 · Recomendações (4:30 – 4:55)
> Recomendamos cinco movimentos: rodar o modelo a cada PEDE e agir antes dos anos de transição; usar o engajamento como termômetro mensal; reforçar Matemática e Português nas Fases 2 a 4; criar um protocolo de permanência para alunos Quartzo e Ágata; e padronizar a coleta dos indicadores. Todo o código está no nosso GitHub. Obrigado!

---

**Checklist antes de gravar**
- [ ] App publicado e aberto em outra aba (a primeira carga do Streamlit Cloud pode demorar).
- [ ] Apresentação em tela cheia.
- [ ] Cronometrar um ensaio: o limite é 5 minutos.
