"""Pacote do TCC: previsão semanal da entrada bruta de caixa por vendedor (Olist).

Cada módulo corresponde a seções do notebook monolítico v5:

    alvo         seções 1–5      leitura, alvo, auditorias, TEST_END
    exogenas     seção 5.1       SGS/BCB e calendário
    painel       seções 6–11     coorte, painel, atributos, vazamento, divisão
    metricas     seção 12        métricas e escala do MASE
    modelos      seção 13        modelos globais e baselines
    cv           seções 14–16    origens móveis, Ridge, seleção
    avaliacao    seções 17, 20, 22, 25
    inferencia   seções 18–19    bootstrap e Diebold-Mariano (HLN)
    diagnostico  seção 23
    experimento1 seção 24
    secundaria   análise secundária (ARIMA, Prophet, LSTM)
    graficos     figuras
    io           contratos entre notebooks (hash, trava do teste)
"""
from .config import iniciar  # noqa: F401
from .painel import TARGET, Especificacao  # noqa: F401
