
install.packages(c("Matrix", "glmnet"))   # solo si no los tienes; xgboost ya lo usas en la app
sink("salida_MHMisinfo.txt", split = TRUE) # guarda toda la salida en un archivo de texto
source("MHMisinfo_analisis_completo.R")
sink()
# =============================================================================
# MHMisinfo — ANÁLISIS COMPLETO DEL ARTÍCULO (script único, reemplaza a 07 y 08)
# ¿Detectan los proxies DISCERN/HONcode de la app, y los modelos de texto, la
# desinformación en salud mental juzgada por expertos? Metadatos vs. transcripción.
# =============================================================================
# Datos: MHMisinfo-Gold (Nguyen et al., 2025, ICWSM 19): 739 videos cortos anotados
#   por 3 profesionales de salud mental que vieron cada video completo.
# Criterio: etiqueta acumulada de expertos (1 = desinformación). Las etiquetas del
#   conjunto Full (GPT-4/RoBERTa) NUNCA son criterio; solo entrenan en la parte I.
# Principal: YouTube (n = 635); BitChute solo para transferencia (parte G).
# Protocolo: 10 x CV estratificada de 5 folds; TF-IDF y lambda se ajustan SOLO
#   dentro del entrenamiento de cada fold; lambda por DEVIANZA (log-verosimilitud) sobre
#   una secuencia completa (lambda.min.ratio = 1e-6), comprobando que no quede en el borde.
# PARTES: A datos y proxies (siempre) | B validación cruzada por representación |
#   C resumen y contrastes | D calibración | E curva de aprendizaje |
#   F interpretabilidad | G transferencia a BitChute | H XGBoost (si está instalado) |
#   I supervisión débil | J criterios de los expertos (exploratorio)
# Uso: Rscript MHMisinfo_analisis_completo.R            -> todo (en tu equipo)
#      Rscript MHMisinfo_analisis_completo.R B Title    -> parte B, solo "Title"
# =============================================================================

## ---- 0. Paquetes y parámetros ----
for (p in c("Matrix", "glmnet")) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, repos = "https://cloud.r-project.org")
library(Matrix); library(glmnet)
TIENE_XGB <- requireNamespace("xgboost", quietly = TRUE)   # opcional (análisis secundario)

RUTA_GOLD <- "videos_MHMisinfo_Gold.csv"   # ajustar si están en otra carpeta
RUTA_FULL <- "videos_MHMisinfo_Full.csv"
N_REP <- 10; K <- 5; B_BOOT <- 2000
TIPO_LAMBDA <- "deviance"   # criterio para elegir lambda (log-verosimilitud binomial)
LMR <- 1e-6                 # lambda.min.ratio: con el valor por defecto, glmnet (ridge, más variables que
# casos) cortaba la secuencia en ~5 valores y la CV elegía el BORDE, dejando
# un modelo sobrepenalizado con predicciones casi constantes
BORDE <- logical(0)

# Lista estándar de palabras vacías en inglés (318 términos; la misma de scikit-learn)
STOP <- c("a","about","above","across","after","afterwards","again","against","all","almost","alone","along","already","also","although","always","am","among","amongst","amoungst","amount","an","and","another","any","anyhow","anyone","anything","anyway","anywhere","are","around","as","at","back","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","below","beside","besides","between","beyond","bill","both","bottom","but","by","call","can","cannot","cant","co","con","could","couldnt","cry","de","describe","detail","do","done","down","due","during","each","eg","eight","either","eleven","else","elsewhere","empty","enough","etc","even","ever","every","everyone","everything","everywhere","except","few","fifteen","fifty","fill","find","fire","first","five","for","former","formerly","forty","found","four","from","front","full","further","get","give","go","had","has","hasnt","have","he","hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","him","himself","his","how","however","hundred","i","ie","if","in","inc","indeed","interest","into","is","it","its","itself","keep","last","latter","latterly","least","less","ltd","made","many","may","me","meanwhile","might","mill","mine","more","moreover","most","mostly","move","much","must","my","myself","name","namely","neither","never","nevertheless","next","nine","no","nobody","none","noone","nor","not","nothing","now","nowhere","of","off","often","on","once","one","only","onto","or","other","others","otherwise","our","ours","ourselves","out","over","own","part","per","perhaps","please","put","rather","re","same","see","seem","seemed","seeming","seems","serious","several","she","should","show","side","since","sincere","six","sixty","so","some","somehow","someone","something","sometime","sometimes","somewhere","still","such","system","take","ten","than","that","the","their","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","thereupon","these","they","thick","thin","third","this","those","though","three","through","throughout","thru","thus","to","together","too","top","toward","towards","twelve","twenty","two","un","under","until","up","upon","us","very","via","was","we","well","were","what","whatever","when","whence","whenever","where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","whose","why","will","with","within","without","would","yet","you","your","yours","yourself","yourselves")

## ---- 1. Lectura y limpieza ----
leer <- function(ruta) {
  d <- read.csv(ruta, stringsAsFactors = FALSE, encoding = "UTF-8")
  d$video_id <- as.character(d$video_id)
  d <- d[!duplicated(d$video_id), ]                      # 4 duplicados en Gold
  for (c in c("video_title", "video_description", "audio_transcript"))
    d[[c]][is.na(d[[c]])] <- ""
  d$y <- as.integer(d$label == -1)                      # 1 = desinformación
  d
}
gold <- leer(RUTA_GOLD)
yt   <- gold[gold$platform == "Youtube", ]
y    <- yt$y
cat(sprintf("Gold sin duplicados: %d | YouTube: %d | desinformación en YouTube: %d (%.1f%%)\n",
            nrow(gold), nrow(yt), sum(y), 100 * mean(y)))

## ---- 2. Tokenización (unigramas + bigramas tras quitar palabras vacías) ----
tokenizar <- function(txt) {
  tk <- regmatches(tolower(txt), gregexpr("\\b\\w\\w+\\b", tolower(txt), perl = TRUE))
  lapply(tk, function(t) {
    t <- t[!t %in% STOP]
    if (length(t) >= 2) c(t, paste(t[-length(t)], t[-1])) else t
  })
}
tok_meta  <- tokenizar(paste(yt$video_title, yt$video_description))
tok_title <- tokenizar(yt$video_title)
tok_trans <- tokenizar(yt$audio_transcript)

## ---- 3. TF-IDF ajustado SOLO con el entrenamiento ----
# tf sublineal (1 + log tf), idf suavizado log((1+n)/(1+df)) + 1, norma L2 por fila
tfidf <- function(tok_tr, tok_te, min_df = 2) {
  dfc   <- table(unlist(lapply(tok_tr, unique)))
  vocab <- names(dfc)[dfc >= min_df]
  idf   <- log((1 + length(tok_tr)) / (1 + as.numeric(dfc[vocab]))) + 1
  armar <- function(tok) {
    j <- match(unlist(tok), vocab); i <- rep(seq_along(tok), lengths(tok)); ok <- !is.na(j)
    m <- sparseMatrix(i = i[ok], j = j[ok], x = 1, dims = c(length(tok), length(vocab)), dimnames = list(NULL, vocab))
    m@x <- 1 + log(m@x)
    m <- m %*% Diagonal(x = idf)
    rn <- sqrt(rowSums(m^2)); rn[rn == 0] <- 1
    out <- as(Diagonal(x = 1 / rn) %*% m, "CsparseMatrix"); colnames(out) <- vocab; out
  }
  list(tr = armar(tok_tr), te = armar(tok_te))
}
eng <- log1p(as.matrix(yt[, c("video_view_count", "video_like_count", "video_comment_count")]))
eng[is.na(eng)] <- 0

representaciones <- c("Engagement", "Title", "Metadata", "Transcript", "Metadata+Transcript")
construir_X <- function(rep_, tr, te) {
  if (rep_ == "Engagement") return(list(tr = eng[tr, ], te = eng[te, ]))
  fuentes <- switch(rep_, "Title" = list(tok_title), "Metadata" = list(tok_meta),
                    "Transcript" = list(tok_trans), "Metadata+Transcript" = list(tok_meta, tok_trans))
  partes <- lapply(fuentes, function(f) tfidf(f[tr], f[te]))
  list(tr = do.call(cbind, lapply(partes, `[[`, "tr")), te = do.call(cbind, lapply(partes, `[[`, "te")))
}

## ---- 4. Utilidades: folds estratificados y métricas ----
folds_estrat <- function(y, k) {
  f <- integer(length(y))
  for (cl in unique(y)) { idx <- which(y == cl); f[idx] <- sample(rep(seq_len(k), length.out = length(idx))) }
  f
}
auc_roc <- function(y, p) { r <- rank(p); n1 <- sum(y); n0 <- sum(1 - y)
(sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
auc_pr <- function(y, p) {                        # precisión media (average precision)
  o <- order(p, decreasing = TRUE); y <- y[o]; p <- p[o]
  u <- c(which(diff(p) != 0), length(p))
  tp <- cumsum(y)[u]; fp <- cumsum(1 - y)[u]
  sum(diff(c(0, tp / sum(y))) * tp / (tp + fp))
}
kappa <- function(y, z) { po <- mean(y == z); pe <- mean(y) * mean(z) + (1 - mean(y)) * (1 - mean(z)); (po - pe) / (1 - pe) }
metricas <- function(y, p) {
  z <- as.integer(p >= 0.5); tp <- sum(z & y); fp <- sum(z & !y); fn <- sum(!z & y)
  c(AUC_ROC = auc_roc(y, p), AUC_PR = auc_pr(y, p), Brier = mean((p - y)^2),
    BalAcc = mean(c(tp / sum(y), sum(!z & !y) / sum(!y))),
    F1_pos = ifelse(tp == 0, 0, 2 * tp / (2 * tp + fp + fn)), Kappa = kappa(y, z),
    Accuracy = mean(z == y))
}

## ---- 5. Modelos ----
pesos <- function(y) ifelse(y == 1, length(y) / (2 * sum(y)), length(y) / (2 * sum(1 - y)))
logistica <- function(X, yv, Xte, inner, estandarizar) {
  cv <- cv.glmnet(X, yv, family = "binomial", alpha = 0, weights = pesos(yv),
                  type.measure = TIPO_LAMBDA, foldid = inner, standardize = estandarizar,
                  lambda.min.ratio = LMR)
  as.numeric(predict(cv, Xte, s = "lambda.min", type = "response"))
}
xgb_fit <- function(X, yv, Xte) {
  # Configuración DETERMINISTA (sin submuestreo aleatorio) para que los resultados
  # sean idénticos en cualquier equipo
  dtr <- xgboost::xgb.DMatrix(X, label = yv)
  m <- xgboost::xgb.train(params = list(objective = "binary:logistic", max_depth = 3, eta = 0.05,
                                        subsample = 1, colsample_bytree = 1, tree_method = "exact",
                                        scale_pos_weight = sum(1 - yv) / sum(yv), nthread = 1),
                          data = dtr, nrounds = 300, verbose = 0)
  predict(m, xgboost::xgb.DMatrix(Xte))
}

## ---- Partes a ejecutar ----
ARGS   <- commandArgs(trailingOnly = TRUE)
PARTES <- if (length(ARGS) == 0) c("B", "C", "D", "E", "F", "G", "H", "I", "J") else toupper(ARGS[1])
SUB    <- if (length(ARGS) >= 2) ARGS[-1] else NULL        # subconjunto opcional (representación o criterio)

# =============================================================================
# PARTE A — Proxies DISCERN/HONcode de app.R traducidos al inglés (siempre se ejecuta)
# =============================================================================
# Traducción término a término de los diccionarios y reglas de app.R. Dos
# adaptaciones necesarias por el idioma: (1) "OMS/WHO" se busca como "world health
# organization" ("who" es un pronombre frecuente en inglés); (2) palabras cortas
# que coincidirían dentro de otras ("how", "ask", "i am") se buscan como palabra
# completa. Se conservan las peculiaridades de la app (p. ej., "dr." nunca coincide
# porque la limpieza elimina el punto). D2 no llega a 5 porque exige duración >= 5
# minutos y el conjunto de datos no trae duración.
limpiar <- function(x) {
  x <- tolower(ifelse(is.na(x), "", x)); x <- gsub("[\r\n\t]", " ", x)
  x <- gsub("[^[:alnum:][:space:]]", " ", x); trimws(gsub("\\s+", " ", x))
}
det    <- function(txt, patron) grepl(patron, txt, perl = TRUE)
cuenta <- function(txt, palabras) {
  patron <- paste0("\\b(", paste(palabras, collapse = "|"), ")\\b")
  lengths(regmatches(txt, gregexpr(patron, txt, perl = TRUE, ignore.case = TRUE)))
}
DIC <- list(
  clickbait = c("incredible", "shocking", "secret", "miracle", "cure", "never", "cured", "amazing", "miraculous",
                "permanent cure", "eliminate forever", "secret method", "revolutionary", "discovery", "surprising", "unbelievable"),
  profesionales = c("doctor", "psychologist", "psychiatrist", "therapist", "phd", "md", "dr", "specialist", "licensed", "physician"),
  instituciones = c("university", "hospital", "clinic", "world health organization", "research", "study", "institute",
                    "apa", "dsm", "ministry of health", "department of health", "association", "college"),
  evidencia = c("study", "studies", "research", "evidence", "clinical trial", "meta analysis", "journal", "published",
                "scientific", "reviewed", "peer reviewed"),
  sintomas = c("sadness", "anhedonia", "hopelessness", "fatigue", "insomnia", "loss of interest", "guilt", "anxiety", "panic",
               "stress", "phobia", "obsession", "compulsion", "trauma", "hallucination", "delusion", "depression", "bipolar", "schizophrenia"),
  tratamiento = c("antidepressant", "antidepressants", "therapy", "psychotherapy", "medication", "cognitive behavioral", "cbt",
                  "mindfulness", "anxiolytic", "antipsychotic", "emdr", "exposure therapy", "drug", "medicine", "treatment"),
  riesgo_suicida = c("suicide", "suicidal", "self harm", "take your own life", "emergency", "crisis", "helpline", "hotline",
                     "prevention", "call", "urgent"),
  pseudociencia = c("positive energy", "vibration", "vibrations", "quantum", "chakra", "law of attraction", "biomagnetism",
                    "healing", "spiritual cleansing", "pseudoscience", "conspiracy"))

proxies_en <- function(titulo, descripcion, transcripcion = NULL) {
  base <- paste(titulo, descripcion); if (!is.null(transcripcion)) base <- paste(base, transcripcion)
  t  <- limpiar(base); ti <- limpiar(titulo)
  clickbait <- cuenta(t, DIC$clickbait) > 0; prof <- cuenta(t, DIC$profesionales) > 0
  inst <- cuenta(t, DIC$instituciones) > 0; estudios <- cuenta(t, DIC$evidencia) > 0
  trat <- cuenta(t, DIC$tratamiento) > 0;   sint2 <- cuenta(t, DIC$sintomas) >= 2
  suic <- cuenta(t, DIC$riesgo_suicida) > 0; pseudo <- cuenta(t, DIC$pseudociencia) > 0
  objetivo <- det(t, "mental health|depression|anxiety|stress") &
    (det(t, "treatment|symptoms|what is") | det(ti, "mental health|treatment|symptoms"))
  D1 <- ifelse(objetivo & !clickbait, 5, ifelse(objetivo, 4, 2))
  D2 <- ifelse(det(ti, "treatment|\\bhow\\b|cure") & trat, 4, 3)
  D3 <- ifelse(sint2 & det(t, "seek help|get help|consult|professional") & trat, 5, ifelse(sint2 | trat, 4, 2))
  pts <- estudios + prof + inst; D4 <- ifelse(pts >= 2, 5, ifelse(pts == 1, 4, 2))
  D5 <- ifelse(det(t, "current|recent|20[2-9][0-9]"), 5, 3)
  demoniza <- det(t, "pills are bad|don t take antidepressants|do not take antidepressants")
  D6 <- ifelse(!pseudo & !demoniza & !clickbait, 5, ifelse(!pseudo & !demoniza, 4, 2))
  linea <- det(t, "helpline|hotline|emergency|crisis"); recursos <- det(t, "more information|resources|links")
  D7 <- ifelse(linea & recursos, 5, ifelse(linea | recursos, 4, 2))
  indiv <- det(t, "depends|varies|each person|consult"); nocura <- !det(t, "permanent cure|eliminate|guaranteed")
  D8 <- ifelse(indiv & nocura, 5, ifelse(indiv | nocura, 4, 2))
  T9  <- ifelse(det(t, "works|acts|mechanism|neurotransmitter|process"), 5, 3)
  T10 <- ifelse(det(t, "benefit|improve|reduce|relieve"), 5, 3)
  efect <- det(t, "side effect|risk|adverse"); superv <- det(t, "supervision|under control|prescription")
  T11 <- ifelse(efect & superv, 5, ifelse(efect | superv, 4, 2))
  cron <- det(t, "chronic|worsens|gets worse|without treatment|untreated")
  T12 <- ifelse(suic & cron, 5, ifelse(suic | cron, 4, 2))
  T13 <- ifelse(det(t, "work|family|relationships|daily life"), 5, 3)
  opc <- det(t, "antidepressant|medication") + det(t, "psychotherapy|therapy") + det(t, "exercise|lifestyle")
  T14 <- ifelse(opc >= 2, 5, ifelse(opc == 1, 4, 2))
  consulta <- det(t, "\\bask\\b|consult|talk to"); urg <- suic | det(t, "emergency|crisis")
  T15 <- ifelse(consulta & urg, 5, ifelse(urg, 4, ifelse(consulta, 3, 2)))
  aj  <- pmax(0, D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + T9 + T10 + T11 + T12 + T13 + T14 + T15 -
                (5 * (pseudo & !prof) + 3 * (!suic) + 3 * demoniza))
  G16 <- ifelse(aj >= 50, 5, ifelse(aj >= 35, 4, ifelse(aj >= 20, 3, ifelse(aj >= 10, 2, 1))))
  cred <- det(t, "dr\\.|psychiatrist|psychologist")
  H1 <- ifelse(cred & prof, 2, ifelse(cred | prof, 1.5, 0.5))
  H2 <- (det(t, "does not replace|doesn t replace|not a substitute|consult a professional|seek help") +
           (!det(t, "you have depression|you are depressed|you re depressed") | det(t, "only a professional can diagnose"))) * 0.95
  H3 <- ((!det(t, "my patient [A-Z]") | det(t, "anonymous|fictional|example")) + !det(t, "leave your details|sign up|register")) * 0.95
  H4 <- (estudios + det(t, "20[0-9]{2}|recent|current")) * 0.95
  H5 <- (det(t, "advantages|disadvantages|benefits|risks") + !pseudo) * 0.95
  H6 <- (det(t, "contact|email|@|instagram") + det(t, "\\bi am\\b|\\bi m\\b|my name is|dr\\.")) * 0.95
  H7 <- (det(t, "conflict of interest|sponsored|sponsorship|no conflict") + !det(t, "buy|course|discount|my program")) * 0.95
  H8 <- ifelse(!det(t, "link|discount") | det(t, "sponsor:|ad:"), 1.9, 0.9)
  as.matrix(data.frame(D1, D2, D3, D4, D5, D6, D7, D8, T9, T10, T11, T12, T13, T14, T15, G16, H1, H2, H3, H4, H5, H6, H7, H8))
}
PX_M  <- proxies_en(yt$video_title, yt$video_description)
PX_MT <- proxies_en(yt$video_title, yt$video_description, yt$audio_transcript)
cat(sprintf("Proxies: perfiles distintos con metadatos = %d; con metadatos + transcripción = %d (de %d videos)\n",
            nrow(unique(PX_M)), nrow(unique(PX_MT)), nrow(PX_M)))
cat(sprintf("Confusión por plataforma: AUC-ROC usando solo 'es BitChute' = %.3f (Gold, n = %d)\n",
            auc_roc(gold$y, as.integer(gold$platform == "Bitchute")), nrow(gold)))

set.seed(2025)                                                  # folds externos (idénticos en todas las partes)
externos <- lapply(seq_len(N_REP), function(r) folds_estrat(y, K))
write.csv(cbind(PX_MT, y = y), "mh_proxies_metadatos_transcripcion.csv", row.names = FALSE)
write.csv(cbind(PX_M, y = y), "mh_proxies_metadatos.csv", row.names = FALSE)
write.csv(as.data.frame(setNames(externos, paste0("rep", seq_len(N_REP)))), "mh_folds.csv", row.names = FALSE)

REPRESENTACIONES <- c("Engagement", "Title", "Metadata", "Transcript", "Metadata+Transcript",
                      "Proxies (metadata)", "Proxies (metadata + transcript)")
archivo_oof <- function(rep_, pond = TRUE) paste0("oof_", gsub("[^A-Za-z]", "", rep_), ifelse(pond, "", "_sinpesos"), ".rds")

# Logística ridge (lambda por devianza) que devuelve además un umbral elegido SOLO con
# el entrenamiento: Youden sobre las predicciones internas fuera de fold (fit.preval
# de glmnet viene en escala logit, por eso se transforma con plogis).
logistica_umbral <- function(X, yv, Xte, inner, estandarizar, ponderar = TRUE) {
  w  <- if (ponderar) pesos(yv) else rep(1, length(yv))
  cv <- cv.glmnet(X, yv, family = "binomial", alpha = 0, weights = w, type.measure = TIPO_LAMBDA,
                  foldid = inner, standardize = estandarizar, keep = TRUE, lambda.min.ratio = LMR)
  jmin <- which(cv$lambda == cv$lambda.min)
  BORDE <<- c(BORDE, jmin == length(cv$lambda))                 # control: lambda elegido en el borde de la secuencia
  p_in <- plogis(cv$fit.preval[, jmin])
  cand <- sort(unique(round(p_in, 4)))
  J <- vapply(cand, function(u) { z <- p_in >= u; mean(z[yv == 1]) + mean(!z[yv == 0]) - 1 }, numeric(1))
  list(p = as.numeric(predict(cv, Xte, s = "lambda.min", type = "response")), umbral = cand[which.max(J)], cv = cv)
}
construir_rep <- function(rep_, tr, te) switch(rep_,
                                               "Proxies (metadata)" = list(tr = PX_M[tr, ], te = PX_M[te, ]),
                                               "Proxies (metadata + transcript)" = list(tr = PX_MT[tr, ], te = PX_MT[te, ]),
                                               construir_X(rep_, tr, te))
es_denso <- function(rep_) grepl("^Proxies|^Engagement", rep_)
cv_repetida <- function(rep_, ponderar = TRUE, reps = seq_len(N_REP)) {
  P <- Z <- matrix(NA_real_, N_REP, length(y))
  for (r in reps) for (k in seq_len(K)) {
    tr <- which(externos[[r]] != k); te <- which(externos[[r]] == k)
    X <- construir_rep(rep_, tr, te)
    set.seed(1000 * r + k); inner <- folds_estrat(y[tr], 5)        # mismos folds internos en todas las representaciones
    res <- logistica_umbral(X$tr, y[tr], X$te, inner, estandarizar = es_denso(rep_), ponderar = ponderar)
    P[r, te] <- res$p; Z[r, te] <- as.numeric(res$p >= res$umbral)
  }
  list(P = P, Z = Z)
}
resumen_rep <- function(P, reps = seq_len(N_REP)) {
  m <- t(sapply(reps, function(r) metricas(y, P[r, ]))); data.frame(media = colMeans(m), de = apply(m, 2, sd))
}
resumen_conf <- function(Z, reps = seq_len(N_REP)) {
  f <- t(sapply(reps, function(r) { z <- Z[r, ]
  tp <- sum(z == 1 & y == 1); fp <- sum(z == 1 & y == 0); fn <- sum(z == 0 & y == 1); tn <- sum(z == 0 & y == 0)
  c(VP = tp, FP = fp, FN = fn, VN = tn, Sensibilidad = tp / (tp + fn), Especificidad = tn / (tn + fp),
    Precision = tp / (tp + fp), F1 = 2 * tp / (2 * tp + fp + fn), BalAcc = (tp / (tp + fn) + tn / (tn + fp)) / 2) }))
  data.frame(media = colMeans(f), de = apply(f, 2, sd))
}
dif_media <- function(i, A, Bm, f) mean(sapply(seq_len(N_REP), function(r) f(y[i], A[r, i]) - f(y[i], Bm[r, i])))

# =============================================================================
# PARTE B — Validación cruzada repetida por representación (guarda las predicciones)
# =============================================================================
if ("B" %in% PARTES) {
  cat(sprintf("\n=== PARTE B (lambda por %s) ===\nClases por etapa: entrenamiento ~%d videos (~%d desinformación); prueba ~%d (~%d)\n",
              TIPO_LAMBDA, round(length(y) * .8), round(sum(y) * .8), round(length(y) * .2), round(sum(y) * .2)))
  for (rep_ in (if (is.null(SUB)) REPRESENTACIONES else SUB)) {
    BORDE <- logical(0); R <- cv_repetida(rep_); saveRDS(R, archivo_oof(rep_))
    cat(sprintf("   control: lambda en el borde de la secuencia en %d de %d ajustes\n", sum(BORDE), length(BORDE)))
    cat("\n--", rep_, "--\n"); print(round(cbind(resumen_rep(R$P)), 3))
  }
}

# =============================================================================
# PARTE C — Tabla resumen, umbral anidado y contrastes pareados por bootstrap
# =============================================================================
if ("C" %in% PARTES) {
  OOF <- lapply(setNames(REPRESENTACIONES, REPRESENTACIONES), function(r) if (file.exists(archivo_oof(r))) readRDS(archivo_oof(r)))
  OOF <- OOF[!sapply(OOF, is.null)]
  tabla <- do.call(rbind, lapply(names(OOF), function(r) {
    a <- resumen_rep(OOF[[r]]$P); b <- resumen_conf(OOF[[r]]$Z); b5 <- resumen_conf(1 * (OOF[[r]]$P >= 0.5))
    data.frame(representacion = r,
               AUC_ROC = sprintf("%.3f (%.3f)", a["AUC_ROC", 1], a["AUC_ROC", 2]), AUC_PR = sprintf("%.3f (%.3f)", a["AUC_PR", 1], a["AUC_PR", 2]),
               Brier = sprintf("%.3f (%.3f)", a["Brier", 1], a["Brier", 2]),
               F1_05 = sprintf("%.3f", a["F1_pos", 1]), Kappa_05 = sprintf("%.3f", a["Kappa", 1]),
               Sens_05 = sprintf("%.3f", b5["Sensibilidad", 1]), Espec_05 = sprintf("%.3f", b5["Especificidad", 1]),
               Prec_05 = sprintf("%.3f", b5["Precision", 1]), VP_05 = sprintf("%.1f", b5["VP", 1]),
               FP_05 = sprintf("%.1f", b5["FP", 1]), FN_05 = sprintf("%.1f", b5["FN", 1]), VN_05 = sprintf("%.1f", b5["VN", 1]),
               Sens_anidado = sprintf("%.3f (%.3f)", b["Sensibilidad", 1], b["Sensibilidad", 2]),
               Espec_anidado = sprintf("%.3f (%.3f)", b["Especificidad", 1], b["Especificidad", 2]),
               Prec_anidado = sprintf("%.3f", b["Precision", 1]), F1_anidado = sprintf("%.3f", b["F1", 1]),
               VP = sprintf("%.1f", b["VP", 1]), FP = sprintf("%.1f", b["FP", 1]), FN = sprintf("%.1f", b["FN", 1]), VN = sprintf("%.1f", b["VN", 1]))
  }))
  cat("\n=== PARTE C: resumen (media (DE) en 10 repeticiones) ===\n"); print(tabla, row.names = FALSE)
  write.csv(tabla, "mh_tabla_principal.csv", row.names = FALSE)
  contrastes <- list(c("Transcript", "Metadata"), c("Metadata+Transcript", "Metadata"),
                     c("Proxies (metadata + transcript)", "Proxies (metadata)"),
                     c("Metadata", "Proxies (metadata)"), c("Metadata+Transcript", "Proxies (metadata + transcript)"))
  contrastes <- contrastes[sapply(contrastes, function(cc) all(cc %in% names(OOF)))]
  set.seed(2025); idx_boot <- replicate(B_BOOT, sample.int(length(y), replace = TRUE), simplify = FALSE)
  tb <- do.call(rbind, lapply(contrastes, function(cc) {
    A <- OOF[[cc[1]]]$P; Bm <- OOF[[cc[2]]]$P
    d <- sapply(idx_boot, function(i) if (sum(y[i]) == 0) c(NA, NA) else c(dif_media(i, A, Bm, auc_roc), dif_media(i, A, Bm, auc_pr)))
    todos <- seq_along(y)
    data.frame(contraste = paste(cc[1], "-", cc[2]),
               dAUC_ROC = dif_media(todos, A, Bm, auc_roc), ROC_inf = quantile(d[1, ], .025, na.rm = TRUE), ROC_sup = quantile(d[1, ], .975, na.rm = TRUE),
               dAUC_PR  = dif_media(todos, A, Bm, auc_pr),  PR_inf  = quantile(d[2, ], .025, na.rm = TRUE), PR_sup  = quantile(d[2, ], .975, na.rm = TRUE))
  }))
  cat("\n=== Contrastes pareados (bootstrap, IC 95%) ===\n"); print(tb, digits = 3, row.names = FALSE)
  write.csv(tb, "mh_contrastes.csv", row.names = FALSE)
}

# =============================================================================
# PARTE D — Calibración: con pesos balanceados vs. sin pesos (metadatos + transcripción)
# =============================================================================
if ("D" %in% PARTES) {
  calib <- function(P) { m <- t(sapply(seq_len(N_REP), function(r) {
    p <- pmin(pmax(P[r, ], 1e-6), 1 - 1e-6)
    c(AUC_ROC = auc_roc(y, p), AUC_PR = auc_pr(y, p), Brier = mean((p - y)^2), Media_p = mean(p),
      Intercepto = unname(coef(glm(y ~ offset(qlogis(p)), family = binomial))[1]),
      Pendiente = unname(coef(glm(y ~ qlogis(p), family = binomial))[2])) }))
  data.frame(media = colMeans(m), de = apply(m, 2, sd)) }
  Rw <- if (file.exists(archivo_oof("Metadata+Transcript"))) readRDS(archivo_oof("Metadata+Transcript")) else cv_repetida("Metadata+Transcript")
  Ru <- cv_repetida("Metadata+Transcript", ponderar = FALSE); saveRDS(Ru, archivo_oof("Metadata+Transcript", FALSE))
  cat("\n=== PARTE D: calibración (metadatos + transcripción) ===\nCon pesos balanceados:\n"); print(round(calib(Rw$P), 3))
  cat("Sin pesos:\n"); print(round(calib(Ru$P), 3))
  cat(sprintf("Brier de un pronóstico constante igual a la prevalencia: %.3f\n", mean(y) * (1 - mean(y))))
}

# =============================================================================
# PARTE E — Curva de aprendizaje (3 repeticiones de la CV externa)
# =============================================================================
if ("E" %in% PARTES) {
  filas_e <- list()
  for (rep_ in (if (is.null(SUB)) c("Metadata", "Transcript", "Metadata+Transcript") else SUB)) for (m in c(100, 200, 300, 400, NA)) {
    P <- matrix(NA_real_, 3, length(y))
    for (r in 1:3) for (k in seq_len(K)) {
      tr_all <- which(externos[[r]] != k); te <- which(externos[[r]] == k)
      if (is.na(m)) tr <- tr_all else {
        set.seed(5000 * r + 10 * k + m); pos <- tr_all[y[tr_all] == 1]; neg <- tr_all[y[tr_all] == 0]
        np <- round(m * length(pos) / length(tr_all)); tr <- c(sample(pos, np), sample(neg, m - np))
      }
      X <- construir_rep(rep_, tr, te); set.seed(1000 * r + k); inner <- folds_estrat(y[tr], 5)
      P[r, te] <- logistica_umbral(X$tr, y[tr], X$te, inner, estandarizar = es_denso(rep_))$p
    }
    roc <- sapply(1:3, function(r) auc_roc(y, P[r, ])); pr <- sapply(1:3, function(r) auc_pr(y, P[r, ]))
    filas_e[[length(filas_e) + 1]] <- data.frame(representacion = rep_, n_entrenamiento = ifelse(is.na(m), 508, m),
                                                 AUC_ROC = mean(roc), AUC_ROC_de = sd(roc), AUC_PR = mean(pr), AUC_PR_de = sd(pr))
  }
  curva <- do.call(rbind, filas_e); cat("\n=== PARTE E: curva de aprendizaje ===\n"); print(curva, digits = 3, row.names = FALSE)
  write.csv(curva, paste0("mh_curva_", gsub("[^A-Za-z]", "", paste(unique(curva$representacion), collapse = "")), ".csv"), row.names = FALSE)
}

# =============================================================================
# PARTE F — Interpretabilidad (modelos ajustados con los 635 videos)
# =============================================================================
if ("F" %in% PARTES) {
  set.seed(99); inner_all <- folds_estrat(y, 5)
  pm <- tfidf(tok_meta, tok_meta); pt <- tfidf(tok_trans, tok_trans)
  colnames(pm$tr) <- paste0("[M] ", colnames(pm$tr)); colnames(pt$tr) <- paste0("[T] ", colnames(pt$tr))
  cvf <- logistica_umbral(cbind(pm$tr, pt$tr), y, cbind(pm$tr, pt$tr)[1:2, ], inner_all, estandarizar = FALSE)$cv
  b <- coef(cvf, s = "lambda.min")[-1, 1]; top <- data.frame(termino = names(b), coef = as.numeric(b))
  cat("\n=== PARTE F ===\n-- 15 términos más asociados a DESINFORMACIÓN --\n"); print(head(top[order(-top$coef), ], 15), row.names = FALSE, digits = 3)
  cat("-- 15 términos más asociados a NO desinformación --\n"); print(head(top[order(top$coef), ], 15), row.names = FALSE, digits = 3)
  write.csv(top[order(-top$coef), ], "mh_coeficientes_terminos.csv", row.names = FALSE)
  cvp <- logistica_umbral(PX_MT, y, PX_MT[1:2, ], inner_all, estandarizar = TRUE)$cv
  bp <- coef(cvp, s = "lambda.min")[-1, 1] * apply(PX_MT, 2, sd)     # log-odds por 1 DE del proxy
  cat("-- Proxies (metadatos + transcripción): log-odds por 1 DE --\n"); print(round(sort(bp, decreasing = TRUE), 3))
  write.csv(data.frame(proxy = names(bp), logodds_por_DE = bp), "mh_coeficientes_proxies.csv", row.names = FALSE)
}

# =============================================================================
# PARTE G — Transferencia entre plataformas: entrenar en YouTube, evaluar en BitChute
# =============================================================================
if ("G" %in% PARTES) {
  bc <- gold[gold$platform == "Bitchute", ]; yb <- bc$y
  tb_meta <- tokenizar(paste(bc$video_title, bc$video_description)); tb_tran <- tokenizar(bc$audio_transcript)
  PXB_MT <- proxies_en(bc$video_title, bc$video_description, bc$audio_transcript)
  set.seed(99); inner_all <- folds_estrat(y, 5)
  ev <- function(Xtr, Xte, dens) logistica_umbral(Xtr, y, Xte, inner_all, estandarizar = dens)$p
  preds <- list(
    "Metadata" = { a <- tfidf(tok_meta, tb_meta); ev(a$tr, a$te, FALSE) },
    "Transcript" = { a <- tfidf(tok_trans, tb_tran); ev(a$tr, a$te, FALSE) },
    "Metadata+Transcript" = { a <- tfidf(tok_meta, tb_meta); b2 <- tfidf(tok_trans, tb_tran); ev(cbind(a$tr, b2$tr), cbind(a$te, b2$te), FALSE) },
    "Proxies (metadata + transcript)" = ev(PX_MT, PXB_MT, TRUE))
  set.seed(2025)
  tg <- do.call(rbind, lapply(names(preds), function(nm) { p <- preds[[nm]]
  bt <- replicate(B_BOOT, { i <- sample.int(length(yb), replace = TRUE); if (sum(yb[i]) %in% c(0, length(i))) NA else auc_roc(yb[i], p[i]) })
  data.frame(representacion = nm, AUC_ROC = auc_roc(yb, p), ic_inf = quantile(bt, .025, na.rm = TRUE),
             ic_sup = quantile(bt, .975, na.rm = TRUE), AUC_PR = auc_pr(yb, p)) }))
  cat(sprintf("\n=== PARTE G: YouTube (n = %d) -> BitChute (n = %d; desinformación = %d, %.1f%%) ===\n",
              length(y), nrow(bc), sum(yb), 100 * mean(yb)))
  print(tg, digits = 3, row.names = FALSE); write.csv(tg, "mh_transferencia_bitchute.csv", row.names = FALSE)
}

# =============================================================================
# PARTE H — XGBoost (el modelo de la app) con los MISMOS folds; requiere el paquete xgboost
# =============================================================================
if ("H" %in% PARTES && TIENE_XGB) {
  filas_h <- list()
  for (rep_ in (if (is.null(SUB)) REPRESENTACIONES else SUB)) {
    P <- matrix(NA_real_, N_REP, length(y))
    for (r in seq_len(N_REP)) for (k in seq_len(K)) {
      tr <- which(externos[[r]] != k); te <- which(externos[[r]] == k)
      X <- construir_rep(rep_, tr, te); P[r, te] <- xgb_fit(X$tr, y[tr], X$te)
    }
    a <- resumen_rep(P); filas_h[[rep_]] <- data.frame(representacion = rep_, AUC_ROC = a["AUC_ROC", 1], AUC_ROC_de = a["AUC_ROC", 2],
                                                       AUC_PR = a["AUC_PR", 1], AUC_PR_de = a["AUC_PR", 2])
  }
  th <- do.call(rbind, filas_h); cat("\n=== PARTE H: XGBoost ===\n"); print(th, digits = 3, row.names = FALSE)
  write.csv(th, "mh_xgboost.csv", row.names = FALSE)
  dtr <- xgboost::xgb.DMatrix(PX_MT, label = y)                     # SHAP del modelo de la app (proxies)
  mx <- xgboost::xgb.train(params = list(objective = "binary:logistic", max_depth = 3, eta = 0.05, subsample = 1, colsample_bytree = 1,
                                         tree_method = "exact", scale_pos_weight = sum(1 - y) / sum(y), nthread = 1), data = dtr, nrounds = 300, verbose = 0)
  shap <- predict(mx, dtr, predcontrib = TRUE); ms <- sort(colMeans(abs(shap[, colnames(PX_MT)])), decreasing = TRUE)
  cat("-- SHAP medio |valor| (XGBoost sobre proxies con metadatos + transcripción) --\n"); print(round(ms, 3))
  write.csv(data.frame(proxy = names(ms), shap_medio_abs = ms), "mh_shap_proxies.csv", row.names = FALSE)
}

# =============================================================================
# PARTE I — Supervisión débil: entrenar con Full-YouTube (etiquetas de GPT-4), evaluar con expertos
# =============================================================================
# ADVERTENCIA: GPT-4 etiquetó leyendo títulos y transcripciones; favorece a la transcripción.
if ("I" %in% PARTES && file.exists(RUTA_FULL)) {
  full <- leer(RUTA_FULL); full <- full[full$platform == "Youtube", ]; yf <- full$y
  tf_meta <- tokenizar(paste(full$video_title, full$video_description)); tf_tran <- tokenizar(full$audio_transcript)
  set.seed(2025); inner_f <- folds_estrat(yf, 5)
  for (rep_ in (if (is.null(SUB)) c("Metadata", "Transcript", "Metadata+Transcript") else SUB)) {
    ftr <- switch(rep_, "Metadata" = list(tf_meta), "Transcript" = list(tf_tran), "Metadata+Transcript" = list(tf_meta, tf_tran))
    fte <- switch(rep_, "Metadata" = list(tok_meta), "Transcript" = list(tok_trans), "Metadata+Transcript" = list(tok_meta, tok_trans))
    partes <- mapply(function(a, b) tfidf(a, b), ftr, fte, SIMPLIFY = FALSE)
    p <- logistica_umbral(do.call(cbind, lapply(partes, `[[`, "tr")), yf, do.call(cbind, lapply(partes, `[[`, "te")), inner_f, FALSE)$p
    saveRDS(p, paste0("debil_", gsub("[^A-Za-z]", "", rep_), ".rds"))
    cat(sprintf("\n=== PARTE I: Full-YouTube (n = %d, GPT-4) -> expertos (n = %d): %s ===\n", length(yf), length(y), rep_))
    print(round(metricas(y, p), 3))
  }
  pd <- lapply(c(Metadata = "Metadata", Transcript = "Transcript", MT = "MetadataTranscript"),
               function(n) if (file.exists(paste0("debil_", n, ".rds"))) readRDS(paste0("debil_", n, ".rds")))
  deb <- do.call(rbind, lapply(names(pd)[!sapply(pd, is.null)], function(n) data.frame(representacion = n, t(metricas(y, pd[[n]])))))
  if (!is.null(pd$Metadata) && !is.null(pd$Transcript)) {
    set.seed(2025)
    dd <- replicate(B_BOOT, { i <- sample.int(length(y), replace = TRUE)
    c(auc_roc(y[i], pd$Transcript[i]) - auc_roc(y[i], pd$Metadata[i]), auc_pr(y[i], pd$Transcript[i]) - auc_pr(y[i], pd$Metadata[i])) })
    cat(sprintf("Transcript - Metadata: dAUC-ROC = %.3f [%.3f, %.3f]; dAUC-PR = %.3f [%.3f, %.3f]\n",
                auc_roc(y, pd$Transcript) - auc_roc(y, pd$Metadata), quantile(dd[1, ], .025), quantile(dd[1, ], .975),
                auc_pr(y, pd$Transcript) - auc_pr(y, pd$Metadata), quantile(dd[2, ], .025), quantile(dd[2, ], .975)))
  }
  write.csv(deb, "mh_supervision_debil.csv", row.names = FALSE)
}

# =============================================================================
# PARTE J — Criterios de los expertos por separado (EXPLORATORIO)
# =============================================================================
if ("J" %in% PARTES) {
  criterios <- c(label_ioi = "Information on Interventions", label_aoc = "Alignment with Medical Consensus",
                 label_ebt = "Evidence-based Treatment")
  alguno_neg <- rowSums(yt[, names(criterios)] == -1) > 0
  cat(sprintf("\n=== PARTE J ===\nCoherencia subetiquetas-etiqueta: %.1f%% (desinformación sin criterio negativo: %d de %d; no desinformación con alguno: %d de %d)\n",
              100 * mean(alguno_neg == (y == 1)), sum(y == 1 & !alguno_neg), sum(y == 1), sum(y == 0 & alguno_neg), sum(y == 0)))
  y_orig <- y
  for (cr in (if (is.null(SUB)) names(criterios) else SUB)) {
    y <- as.integer(yt[[cr]] == -1); set.seed(3025)
    externos_c <- lapply(seq_len(N_REP), function(r) folds_estrat(y, K)); ext_bak <- externos; externos <- externos_c
    RJ <- 1:5                                                      # 5 repeticiones: análisis exploratorio
    filas_c <- lapply(c("Metadata", "Transcript", "Metadata+Transcript"), function(rep_) {
      P <- cv_repetida(rep_, reps = RJ)$P
      roc <- sapply(RJ, function(r) auc_roc(y, P[r, ])); pr <- sapply(RJ, function(r) auc_pr(y, P[r, ]))
      data.frame(criterio = criterios[[cr]], n_neg = sum(y), prevalencia = mean(y), representacion = rep_,
                 AUC_ROC = mean(roc), AUC_ROC_de = sd(roc), AUC_PR = mean(pr), AUC_PR_de = sd(pr)) })
    externos <- ext_bak; out <- do.call(rbind, filas_c); print(out, digits = 3, row.names = FALSE)
    write.csv(out, paste0("mh_criterio_", cr, ".csv"), row.names = FALSE)
  }
  y <- y_orig
}
cat("\nListo.\n")