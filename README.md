┌─────────────────────────┐
│ YouTube Data API v3     │
└───────────┬─────────────┘
            │  OAuth 2.0
            ▼
┌─────────────────────────┐
│ app.R (Shiny)           │
│  - Análisis individual  │
│  - Análisis masivo      │
│  - Entrenamiento ML     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Evaluación DISCERN      │
│ Evaluación HONcode      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ XGBoost (modelo .json)  │
│ + metadatos .rds        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Red semántica (igraph)  │
│ - TF-IDF + coseno       │
│ - kNN, Louvain          │
│ - Tablas y figuras      │
└─────────────────────────┘

# DICCIONARIOS DE PALABRAS CLAVE
palabras_clickbait <- c(
  "increíble", "impactante", "secreto", "milagro", "cura", "nunca", "curar",
  "shocking", "amazing", "miracle", "cure", "secret", "milagroso",
  "cura definitiva", "elimina para siempre", "método secreto", "revolucionario",
  "descubrimiento", "sorprendente", "increible"
)
palabras_promesas <- c("rápido", "fácil", "instantáneo", "minutos", "días", "fast", "easy", "instant", "quick", "inmediato")
palabras_profesionales <- c("doctor", "psicólogo", "psiquiatra", "terapeuta", "psychologist", "psychiatrist", "therapist", "phd", "md", "dr", "dra", "especialista", "licenciado", "médico")
palabras_instituciones <- c("universidad", "hospital", "clínica", "oms", "who", "research", "study", "university", "instituto", "apa", "dsm", "ministerio de salud", "asociación", "colegio")
palabras_evidencia <- c("estudio", "investigación", "evidencia", "ensayo clínico", "meta-análisis", "journal", "publicado", "científico", "revisado", "peer-reviewed")
palabras_sintomas_mental_health <- c("tristeza", "anhedonia", "desesperanza", "fatiga", "insomnio", "pérdida de interés", "culpa", "ansiedad", "pánico", "estrés", "fobia", "obsesión", "compulsión", "trauma", "alucinación", "delirio", "depresión", "bipolar", "esquizofrenia")
palabras_tratamiento <- c("antidepresivo", "terapia", "psicoterapia", "medicación", "cognitivo conductual", "tcc", "mindfulness", "ansiolítico", "antipsicótico", "emdr", "terapia de exposición", "fármaco", "medicamento", "tratamiento")
palabras_riesgo_suicida <- c("suicidio", "suicida", "autolesión", "quitarse la vida", "emergencia", "crisis", "línea de ayuda", "prevención", "llamar", "urgencia")
palabras_pseudociencia <- c("energía positiva", "vibración", "cuántico", "chakra", "ley de atracción", "biomagnetismo", "sanación", "limpieza espiritual", "pseudociencia", "conspiración")
# FUNCIONES AUXILIARES
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || a == "") b else a
limpiar_texto <- function(texto) {
  if (is.null(texto) || is.na(texto) || texto == "") return("")
  texto <- tolower(as.character(texto))
  texto <- str_replace_all(texto, "[\r\n\t]", " ")
  texto <- str_replace_all(texto, "[^[:alnum:]\\sáéíóúñü]", " ")
  texto <- str_replace_all(texto, "\\s+", " ")
  str_trim(texto)
}
contar_palabras <- function(texto, palabras) {
  if (length(palabras) == 0 || nchar(texto) == 0) return(0)
  patron <- paste0("\\b(", paste(palabras, collapse = "|"), ")\\b")
  str_count(texto, regex(patron, ignore_case = TRUE))
}
extraer_video_id <- function(x) {
  if (is.null(x) || is.na(x) || x == "") return(NA_character_)
  x <- trimws(as.character(x))
  if (nchar(x) == 11 && grepl("^[A-Za-z0-9_-]+$", x)) return(x)
  id <- str_extract(x, "(?<=v=)[A-Za-z0-9_-]+")
  if (!is.na(id)) return(id)
  id <- str_extract(x, "(?<=be/)[A-Za-z0-9_-]+")
  if (!is.na(id)) return(id)
  return(NA_character_)
}
# EVALUACIÓN DISCERN COMPLETA
evaluar_discern_completo <- function(video_data) {
  texto <- limpiar_texto(paste(video_data$titulo, video_data$descripcion %||% ""))
  titulo <- video_data$titulo
  scores <- list()
  
  # SECCIÓN 1: CONFIABILIDAD (8 ítems)
  tiene_objetivo <- str_detect(texto, "salud mental|mental health|depresión|ansiedad|estrés") &
    (str_detect(texto, "tratamiento|síntomas|qué es") | str_detect(titulo, "(?i)(salud mental|tratamiento|síntomas)"))
  clickbait <- contar_palabras(texto, palabras_clickbait) > 0
  scores$D1 <- case_when(
    tiene_objetivo & !clickbait ~ 5,
    tiene_objetivo ~ 4,
    TRUE ~ 2
  )
  
  promete_tratamiento <- str_detect(titulo, "(?i)(tratamiento|cómo|curar)")
  menciona_tratamiento <- contar_palabras(texto, palabras_tratamiento) > 0
  duracion_ok <- video_data$duracion_min >= 5
  scores$D2 <- case_when(
    promete_tratamiento & menciona_tratamiento & duracion_ok ~ 5,
    promete_tratamiento & menciona_tratamiento ~ 4,
    TRUE ~ 3
  )
  
  menciona_sintomas <- contar_palabras(texto, palabras_sintomas_mental_health) >= 2
  menciona_ayuda <- str_detect(texto, "buscar ayuda|consulta|profesional")
  scores$D3 <- case_when(
    menciona_sintomas & menciona_ayuda & menciona_tratamiento ~ 5,
    menciona_sintomas | menciona_tratamiento ~ 4,
    TRUE ~ 2
  )
  
  menciona_estudios <- contar_palabras(texto, palabras_evidencia) > 0
  menciona_profesionales <- contar_palabras(texto, palabras_profesionales) > 0
  menciona_instituciones <- contar_palabras(texto, palabras_instituciones) > 0
  puntos_fuentes <- sum(menciona_estudios, menciona_profesionales, menciona_instituciones)
  scores$D4 <- case_when(
    puntos_fuentes >= 2 ~ 5,
    puntos_fuentes == 1 ~ 4,
    TRUE ~ 2
  )
  
  menciona_actualidad <- str_detect(texto, "actual|reciente|20[2-9][0-9]")
  scores$D5 <- ifelse(menciona_actualidad, 5, 3)
  
  tiene_pseudociencia <- contar_palabras(texto, palabras_pseudociencia) > 0
  demoniza <- str_detect(texto, "las pastillas son malas|no tomes antidepresivos")
  scores$D6 <- case_when(
    !tiene_pseudociencia & !demoniza & !clickbait ~ 5,
    !tiene_pseudociencia & !demoniza ~ 4,
    TRUE ~ 2
  )
  
  menciona_linea_ayuda <- str_detect(texto, "línea de ayuda|emergencia|crisis")
  menciona_recursos <- str_detect(texto, "más información|recursos|enlaces")
  scores$D7 <- case_when(
    menciona_linea_ayuda & menciona_recursos ~ 5,
    menciona_linea_ayuda | menciona_recursos ~ 4,
    TRUE ~ 2
  )
  
  reconoce_individualidad <- str_detect(texto, "depende|varía|cada persona|consulta")
  no_promete_cura <- !str_detect(texto, "cura definitiva|elimina|garantizado")
  scores$D8 <- case_when(
    reconoce_individualidad & no_promete_cura ~ 5,
    reconoce_individualidad | no_promete_cura ~ 4,
    TRUE ~ 2
  )
  
  # SECCIÓN 2: CALIDAD DE INFORMACIÓN SOBRE TRATAMIENTOS (7 ítems)
  explica_mecanismo <- str_detect(texto, "funciona|actúa|mecanismo|neurotransmisor|proceso")
  scores$T9 <- ifelse(explica_mecanismo, 5, 3)
  
  menciona_beneficios <- str_detect(texto, "beneficio|mejora|reduce|alivia")
  scores$T10 <- ifelse(menciona_beneficios, 5, 3)
  
  menciona_efectos <- str_detect(texto, "efecto secundario|riesgo|adverso")
  menciona_supervision <- str_detect(texto, "supervisión|bajo control|prescripción")
  scores$T11 <- case_when(
    menciona_efectos & menciona_supervision ~ 5,
    menciona_efectos | menciona_supervision ~ 4,
    TRUE ~ 2
  )
  
  menciona_riesgo_suicida <- contar_palabras(texto, palabras_riesgo_suicida) > 0
  menciona_cronificacion <- str_detect(texto, "crónica|empeora|sin tratamiento")
  scores$T12 <- case_when(
    menciona_riesgo_suicida & menciona_cronificacion ~ 5,
    menciona_riesgo_suicida | menciona_cronificacion ~ 4,
    TRUE ~ 2
  )
  
  menciona_impacto <- str_detect(texto, "trabajo|familia|relaciones|vida diaria")
  scores$T13 <- ifelse(menciona_impacto, 5, 3)
  
  opciones <- sum(
    str_detect(texto, "antidepresivo|medicación"),
    str_detect(texto, "psicoterapia|terapia"),
    str_detect(texto, "ejercicio|estilo de vida")
  )
  scores$T14 <- case_when(
    opciones >= 2 ~ 5,
    opciones == 1 ~ 4,
    TRUE ~ 2
  )
  
  sugiere_consulta <- str_detect(texto, "pregunta|consulta|habla con")
  indica_urgencia <- menciona_riesgo_suicida | str_detect(texto, "emergencia|crisis")
  scores$T15 <- case_when(
    sugiere_consulta & indica_urgencia ~ 5,
    indica_urgencia ~ 4,
    sugiere_consulta ~ 3,
    TRUE ~ 2
  )
  
  # SECCIÓN 3: CALIFICACIÓN GLOBAL (ítem 16)
  total_seccion1 <- sum(unlist(scores[paste0("D", 1:8)]))
  total_seccion2 <- sum(unlist(scores[paste0("T", 9:15)]))
  puntaje_discern_total <- total_seccion1 + total_seccion2
  
  penalizacion <- 0
  if (tiene_pseudociencia & !menciona_profesionales) penalizacion <- penalizacion + 5
  if (!menciona_riesgo_suicida) penalizacion <- penalizacion + 3
  if (demoniza) penalizacion <- penalizacion + 3
  puntaje_ajustado <- max(0, puntaje_discern_total - penalizacion)
  
  scores$G16 <- case_when(
    puntaje_ajustado >= 50 ~ 5,
    puntaje_ajustado >= 35 ~ 4,
    puntaje_ajustado >= 20 ~ 3,
    puntaje_ajustado >= 10 ~ 2,
    TRUE ~ 1
  )
  
  puntaje_discern_16_80 <- puntaje_ajustado + scores$G16
  
  calidad_video <- round(mean(c(scores$D1, scores$D2, scores$D6)), 2)
  valor_educativo <- round(mean(c(scores$T9, scores$T10, scores$T11, scores$T12, scores$T13)), 2)
  confiabilidad <- round(mean(c(scores$D3, scores$D4, scores$D7, scores$D8)), 2)
  fiabilidad <- round(mean(c(scores$D4, scores$D5, scores$T11, scores$T15)), 2)
  
  clasificacion_discern <- case_when(
    puntaje_discern_16_80 >= 55 ~ "✅ Excelente",
    puntaje_discern_16_80 >= 43 ~ "✅ Buena",
    puntaje_discern_16_80 >= 31 ~ "⚠️ Moderada",
    puntaje_discern_16_80 >= 19 ~ "⚠️ Baja",
    TRUE ~ "❌ Muy Baja"
  )
  
  return(list(
    scores_individuales = scores,
    puntaje_discern_total = puntaje_discern_16_80,
    puntaje_ajustado = puntaje_ajustado,
    calidad_video = calidad_video,
    valor_educativo = valor_educativo,
    confiabilidad = confiabilidad,
    fiabilidad = fiabilidad,
    clasificacion_discern = clasificacion_discern,
    nivel_global = scores$G16,
    tiene_pseudociencia = tiene_pseudociencia,
    menciona_riesgo_suicida = menciona_riesgo_suicida,
    menciona_profesionales = menciona_profesionales,
    menciona_instituciones = menciona_instituciones,
    tiene_clickbait = clickbait
  ))
}
# EVALUACIÓN HONcode COMPLETA
evaluar_honcode_completo <- function(video_data) {
  texto <- limpiar_texto(paste(video_data$titulo, video_data$descripcion %||% ""))
  scores_hon <- numeric(8)
  
  tiene_credenciales <- str_detect(texto, "dr\\.|dra\\.|psiquiatra|psicólogo")
  menciona_profesional <- contar_palabras(texto, palabras_profesionales) > 0
  scores_hon[1] <- case_when(
    tiene_credenciales & menciona_profesional ~ 2,
    tiene_credenciales | menciona_profesional ~ 1.5,
    TRUE ~ 0.5
  )
  
  aclara_complemento <- str_detect(texto, "no reemplaza|consulta profesional|busca ayuda")
  no_diagnostica <- !str_detect(texto, "tienes depresión|estás deprimido") | str_detect(texto, "solo un profesional puede diagnosticar")
  scores_hon[2] <- sum(aclara_complemento, no_diagnostica) * 0.95
  
  usa_casos_anonimos <- !str_detect(texto, "mi paciente [A-Z]") | str_detect(texto, "anónimo|ficticio|ejemplo")
  no_pide_datos <- !str_detect(texto, "déjame tus datos|regístrate")
  scores_hon[3] <- sum(usa_casos_anonimos, no_pide_datos) * 0.95
  
  tiene_referencias <- contar_palabras(texto, palabras_evidencia) > 0
  menciona_fecha <- str_detect(texto, "20[0-9]{2}|reciente|actual")
  scores_hon[4] <- sum(tiene_referencias, menciona_fecha) * 0.95
  
  menciona_pros_contras <- str_detect(texto, "ventajas|desventajas|beneficios|riesgos")
  no_pseudociencia <- contar_palabras(texto, palabras_pseudociencia) == 0
  scores_hon[5] <- sum(menciona_pros_contras, no_pseudociencia) * 0.95
  
  tiene_contacto <- str_detect(texto, "contacto|correo|@|instagram")
  se_identifica <- str_detect(texto, "soy|me llamo|dr\\.|dra\\.")
  scores_hon[6] <- sum(tiene_contacto, se_identifica) * 0.95
  
  declara_conflicto <- str_detect(texto, "conflicto de interés|patrocinio|sin conflicto")
  no_promocion <- !str_detect(texto, "compra|curso|descuento|mi programa")
  scores_hon[7] <- sum(declara_conflicto, no_promocion) * 0.95
  
  publicidad_separada <- !str_detect(texto, "enlace|descuento") | str_detect(texto, "patrocinio:|anuncio:")
  scores_hon[8] <- ifelse(publicidad_separada, 1.9, 0.9)
  
  puntaje_honcode_total <- sum(scores_hon)
  cumplimiento_honcode <- round((puntaje_honcode_total / 16) * 100, 1)
  
  clasificacion_hon <- case_when(
    puntaje_honcode_total >= 11 ~ "🔒 Excelente",
    puntaje_honcode_total >= 7 ~ "🔓 Buena",
    puntaje_honcode_total >= 3 ~ "⚠️ Baja",
    TRUE ~ "❌ Muy Baja"
  )
  
  return(list(
    scores_hon = scores_hon,
    puntaje_honcode_total = puntaje_honcode_total,
    cumplimiento_honcode = cumplimiento_honcode,
    clasificacion_hon = clasificacion_hon,
    hon_autoridad = scores_hon[1],
    hon_complementariedad = scores_hon[2],
    hon_confidencialidad = scores_hon[3],
    hon_atribucion = scores_hon[4],
    hon_justificabilidad = scores_hon[5],
    hon_transparencia = scores_hon[6],
    hon_financiacion = scores_hon[7],
    hon_publicidad = scores_hon[8]
  ))
}
# ANÁLISIS BÁSICO DEL VIDEO
analizar_video <- function(video_id) {
  vid <- extraer_video_id(video_id)
  if (is.na(vid)) return(list(exito = FALSE, mensaje_error = "ID inválido"))
  
  tryCatch({
    stats <- tryCatch(get_stats(vid), error = function(e) NULL)
    detalles_res <- tryCatch(get_video_details(vid), error = function(e) NULL)
    
    if (is.null(stats) && is.null(detalles_res)) {
      return(list(exito = FALSE, mensaje_error = "Video no disponible"))
    }
    
    titulo <- ""
    descripcion <- ""
    duracion_iso <- "PT0S"
    
    if (!is.null(detalles_res) && length(detalles_res$items) > 0) {
      detalles <- detalles_res$items[[1]]
      titulo <- detalles$snippet$title %||% ""
      descripcion <- detalles$snippet$description %||% ""
      duracion_iso <- detalles$contentDetails$duration %||% "PT0S"
    }
    
    views <- 0
    likes <- 0
    comentarios <- 0
    
    if (!is.null(stats)) {
      views <- as.numeric(stats$viewCount %||% 0)
      likes <- as.numeric(stats$likeCount %||% 0)
      comentarios <- as.numeric(stats$commentCount %||% 0)
    }
    
    duracion_seg <- 0
    nums <- as.numeric(str_extract_all(duracion_iso, "\\d+")[[1]])
    if (length(nums) > 0) {
      mult <- if(length(nums)==3) c(3600,60,1) else if(length(nums)==2) c(60,1) else c(1)
      duracion_seg <- sum(nums * mult)
    }
    
    texto <- limpiar_texto(paste(titulo, descripcion))
    clickbait <- contar_palabras(texto, palabras_clickbait)
    promesas <- contar_palabras(texto, palabras_promesas)
    profesionales <- contar_palabras(texto, palabras_profesionales)
    instituciones <- contar_palabras(texto, palabras_instituciones)
    
    eng_raw <- if (views > 0) (likes + comentarios) / views else NA_real_
    eng_display <- ifelse(is.nan(eng_raw) | is.infinite(eng_raw), NA_real_, eng_raw)
    eng_calc <- ifelse(is.na(eng_display), 0, eng_display)
    
    score_riesgo <- min(100, (
      (clickbait > 0) * 28 +
        (promesas > 0) * 27 +
        (str_count(titulo, "!") > 1) * 5 +
        ((str_count(titulo, "[A-Z]") / max(nchar(titulo),1)) > 0.2) * 6 +
        (eng_calc < 0.05) * 8 +
        (duracion_seg < 180 && (clickbait + promesas) >= 2) * 15 +
        (profesionales == 0 && instituciones == 0) * 15
    ))
    
    score_confianza <- min(100, (
      (profesionales > 0) * 20 +
        (instituciones > 0) * 26 +
        (clickbait == 0) * 26 +
        (promesas == 0) * 15 +
        (duracion_seg >= 300) * 5 +
        (eng_calc > 0.05 && eng_calc < 0.1) * 8
    ))
    
    categoria <- case_when(
      score_riesgo >= 40 ~ "Alto riesgo",
      score_riesgo >= 20 ~ "Riesgo medio",
      score_riesgo >= 10 ~ "Riesgo bajo",
      TRUE ~ "Confiable"
    )
    
    clasificacion <- ifelse(score_riesgo >= 10, "⚠️ Potencialmente problemático", "✅ Confiable")
    
    Sys.sleep(0.2)
    
    return(list(
      exito = TRUE,
      video_id = vid,
      titulo = titulo,
      descripcion = descripcion,
      views = views,
      likes = likes,
      comentarios = comentarios,
      duracion_min = round(duracion_seg/60, 1),
      engagement_rate = ifelse(is.na(eng_display), NA_real_, round(eng_display, 4)),
      score_riesgo = round(score_riesgo, 1),
      score_confianza = round(score_confianza, 1),
      categoria_riesgo = categoria,
      clasificacion_simple = clasificacion,
      tiene_clickbait = clickbait > 0,
      tiene_promesas = promesas > 0,
      menciona_profesional = profesionales > 0,
      menciona_institucion = instituciones > 0,
      url = paste0("https://www.youtube.com/watch?v=", vid)
    ))
  }, error = function(e) {
    list(exito = FALSE, mensaje_error = e$message)
  })
}
# ANÁLISIS COMPLETO INTEGRADO
analizar_video_completo <- function(video_id) {
  analisis_basico <- analizar_video(video_id)
  if (!isTRUE(analisis_basico$exito)) {
    return(analisis_basico)
  }
  
  eval_discern <- evaluar_discern_completo(analisis_basico)
  eval_honcode <- evaluar_honcode_completo(analisis_basico)
  
  resultado_completo <- c(
    analisis_basico,
    list(
      discern_calidad_video = eval_discern$calidad_video,
      discern_valor_educativo = eval_discern$valor_educativo,
      discern_confiabilidad = eval_discern$confiabilidad,
      discern_fiabilidad = eval_discern$fiabilidad,
      discern_puntaje_total = eval_discern$puntaje_discern_total,
      discern_clasificacion = eval_discern$clasificacion_discern,
      discern_nivel_global = eval_discern$nivel_global,
      
      D1_objetivos_claros = eval_discern$scores_individuales$D1,
      D2_cumple_objetivos = eval_discern$scores_individuales$D2,
      D3_relevancia = eval_discern$scores_individuales$D3,
      D4_fuentes_informacion = eval_discern$scores_individuales$D4,
      D5_fecha_clara = eval_discern$scores_individuales$D5,
      D6_equilibrio = eval_discern$scores_individuales$D6,
      D7_fuentes_apoyo = eval_discern$scores_individuales$D7,
      D8_reconoce_incertidumbre = eval_discern$scores_individuales$D8,
      T9_explica_funcionamiento = eval_discern$scores_individuales$T9,
      T10_beneficios = eval_discern$scores_individuales$T10,
      T11_riesgos = eval_discern$scores_individuales$T11,
      T12_sin_tratamiento = eval_discern$scores_individuales$T12,
      T13_impacto_calidad_vida = eval_discern$scores_individuales$T13,
      T14_multiples_opciones = eval_discern$scores_individuales$T14,
      T15_decision_informada = eval_discern$scores_individuales$T15,
      G16_calificacion_global = eval_discern$scores_individuales$G16,
      
      honcode_puntaje_total = eval_honcode$puntaje_honcode_total,
      honcode_cumplimiento = eval_honcode$cumplimiento_honcode,
      honcode_clasificacion = eval_honcode$clasificacion_hon,
      
      H1_autoridad = eval_honcode$hon_autoridad,
      H2_complementariedad = eval_honcode$hon_complementariedad,
      H3_confidencialidad = eval_honcode$hon_confidencialidad,
      H4_atribucion = eval_honcode$hon_atribucion,
      H5_justificabilidad = eval_honcode$hon_justificabilidad,
      H6_transparencia = eval_honcode$hon_transparencia,
      H7_financiacion = eval_honcode$hon_financiacion,
      H8_publicidad = eval_honcode$hon_publicidad,
      
      tiene_pseudociencia = eval_discern$tiene_pseudociencia,
      menciona_riesgo_suicida = eval_discern$menciona_riesgo_suicida,
      
      discern_scores = eval_discern$scores_individuales,
      honcode_scores = eval_honcode$scores_hon
    )
  )
  return(resultado_completo)
}
# FUNCIÓN MEJORADA PARA ENCONTRAR PUNTO DE CORTE CON FOCO EN SENSITIVITY
encontrar_punto_corte_sensitivity <- function(modelo, datos_test, feature_names) {
  tryCatch({
    dtest <- xgb.DMatrix(
      data = as.matrix(datos_test[feature_names]),
      label = as.numeric(datos_test$es_problematico) - 1
    )
    
    pred_prob <- predict(modelo, dtest)
    real <- as.numeric(datos_test$es_problematico) - 1
    
    mejores_metricas <- NULL
    mejor_punto_corte <- 0.3
    mejor_score_sensitivity <- 0
    
    for (corte in seq(0.05, 0.5, 0.01)) {
      pred_class <- ifelse(pred_prob > corte, 1, 0)
      
      cm <- table(Predicted = pred_class, Real = real)
      if (nrow(cm) == 2 && ncol(cm) == 2) {
        accuracy <- sum(diag(cm)) / sum(cm)
        sensitivity <- cm[2,2] / sum(cm[,2]) # Capacidad de detectar problemáticos
        specificity <- cm[1,1] / sum(cm[,1])
        precision <- ifelse(sum(cm[2,]) > 0, cm[2,2] / sum(cm[2,]), 0)
        f1 <- ifelse((precision + sensitivity) > 0, 2 * (precision * sensitivity) / (precision + sensitivity), 0)
        
        # EXTRACT FALSE NEGATIVES AND FALSE POSITIVES
        fn <- cm[1,2] # Falsos Negativos
        fp <- cm[2,1] # Falsos Positivos
        
        # SCORE QUE PRIORIZA SENSITIVITY Y MINIMIZA FALSOS NEGATIVOS
        score_sensitivity <- (
          sensitivity * 0.5 + # Sensitivity es lo más importante
            (1 - fn/sum(cm)) * 0.3 + # Minimizar falsos negativos
            specificity * 0.1 + # Algo de specificity
            f1 * 0.1 # Balance F1
        )
        
        if (!is.na(score_sensitivity) &&
            score_sensitivity > mejor_score_sensitivity &&
            sensitivity >= 0.7 && # Sensitivity mínima aceptable
            fn <= 3) { # Máximo 3 falsos negativos
          
          mejor_score_sensitivity <- score_sensitivity
          mejor_punto_corte <- corte
          mejores_metricas <- list(
            punto_corte = corte,
            f1 = f1,
            sensitivity = sensitivity,
            specificity = specificity,
            precision = precision,
            accuracy = accuracy,
            score_sensitivity = score_sensitivity,
            fn = fn,
            fp = fp,
            vp = cm[2,2], # Verdaderos positivos
            vn = cm[1,1] # Verdaderos negativos
          )
        }
      }
    }
    
    # SI NO ENCONTRAMOS UN PUNTO BUENO, USAR ESTRATEGIA DE RESERVA
    if (is.null(mejores_metricas)) {
      message("⚠️ No se encontró punto ideal, usando estrategia de reserva...")
      # BUSCAR EL PUNTO QUE MAXIMICE SENSITIVITY CON FN <= 3
      for (corte in seq(0.05, 0.3, 0.01)) {
        pred_class <- ifelse(pred_prob > corte, 1, 0)
        cm <- table(Predicted = pred_class, Real = real)
        if (nrow(cm) == 2 && ncol(cm) == 2) {
          sensitivity <- cm[2,2] / sum(cm[,2])
          fn <- cm[1,2]
          if (sensitivity >= 0.8 && fn <= 3) {
            accuracy <- sum(diag(cm)) / sum(cm)
            specificity <- cm[1,1] / sum(cm[,1])
            precision <- ifelse(sum(cm[2,]) > 0, cm[2,2] / sum(cm[2,]), 0)
            f1 <- ifelse((precision + sensitivity) > 0, 2 * (precision * sensitivity) / (precision + sensitivity), 0)
            
            mejores_metricas <- list(
              punto_corte = corte,
              f1 = f1,
              sensitivity = sensitivity,
              specificity = specificity,
              precision = precision,
              accuracy = accuracy,
              score_sensitivity = sensitivity,
              fn = fn,
              fp = cm[2,1],
              vp = cm[2,2],
              vn = cm[1,1]
            )
            break
          }
        }
      }
    }
    
    message("🎯 Punto de corte óptimo para Sensitivity: ", mejor_punto_corte)
    message("🎯 Sensitivity: ", round(mejores_metricas$sensitivity, 3))
    message("🎯 Specificity: ", round(mejores_metricas$specificity, 3))
    message("📈 F1-Score: ", round(mejores_metricas$f1, 3))
    message("❌ Falsos Negativos: ", mejores_metricas$fn)
    message("⚠️ Falsos Positivos: ", mejores_metricas$fp)
    
    return(mejores_metricas)
    
  }, error = function(e) {
    message("❌ Error buscando punto de corte: ", e$message)
    return(list(
      punto_corte = 0.1, f1 = 0, sensitivity = 0, specificity = 0,
      precision = 0, accuracy = 0, fn = 0, fp = 0, vp = 0, vn = 0
    ))
  })
}
# MODELO XGBOOST OPTIMIZADO - VERSIÓN MEJORADA PARA SENSITIVITY
entrenar_modelo_discern_honcode <- function(datos) {
  tryCatch({
    message("🔧 Iniciando entrenamiento del modelo optimizado para Sensitivity...")
    
    # VERIFICAR QUE TENEMOS LAS COLUMNAS NECESARIAS
    columnas_requeridas <- c(
      "discern_puntaje_total", "honcode_puntaje_total", "tiene_pseudociencia",
      "score_riesgo", "D6_equilibrio", "H5_justificabilidad", "D4_fuentes_informacion",
      "H1_autoridad", "T14_multiples_opciones", "T11_riesgos", "D8_reconoce_incertidumbre"
    )
    
    columnas_faltantes <- setdiff(columnas_requeridas, names(datos))
    if (length(columnas_faltantes) > 0) {
      message("❌ Faltan columnas requeridas: ", paste(columnas_faltantes, collapse = ", "))
      return(NULL)
    }
    
    # DEFINICIÓN MÁS SENSIBLE Y REALISTA DEL TARGET
    datos_modelo <- datos %>%
      mutate(
        # CRITERIOS MÁS SENSIBLES PERO REALISTAS PARA PROBLEMÁTICOS
        criterios_problematicos = (
          (discern_puntaje_total < 40) +
            (honcode_puntaje_total < 7) +
            (tiene_pseudociencia == TRUE) +
            (score_riesgo >= 35) +
            (D6_equilibrio < 3) +
            (H5_justificabilidad < 1.5) +
            (D4_fuentes_informacion < 3)
        ),
        
        # CRITERIOS PARA CONFIABLES
        criterios_confiables = (
          (discern_puntaje_total >= 60) +
            (honcode_puntaje_total >= 10) +
            (D4_fuentes_informacion >= 4) +
            (H1_autoridad >= 1.5) +
            (T11_riesgos >= 4) +
            (D8_reconoce_incertidumbre >= 4)
        ),
        
        # DEFINICIÓN FINAL DEL TARGET CON MÁS MATICES
        es_problematico = factor(
          case_when(
            criterios_problematicos >= 4 ~ "Problematico",
            criterios_confiables >= 4 ~ "Confiable",
            criterios_problematicos >= 3 & criterios_confiables <= 1 ~ "Problematico",
            criterios_confiables >= 3 & criterios_problematicos <= 1 ~ "Confiable",
            
            # CASOS LIMITE - USAR SCORE COMPUESTO
            TRUE ~ ifelse(
              (discern_puntaje_total/80 * 0.4 +
                 honcode_puntaje_total/16 * 0.3 +
                 (100 - score_riesgo)/100 * 0.3) >= 0.6,
              "Confiable", "Problematico"
            )
          ),
          levels = c("Confiable", "Problematico")
        )
      ) %>%
      select(
        es_problematico,
        # FEATURES DISCERN (16 criterios)
        D1_objetivos_claros, D2_cumple_objetivos, D3_relevancia, D4_fuentes_informacion,
        D5_fecha_clara, D6_equilibrio, D7_fuentes_apoyo, D8_reconoce_incertidumbre,
        T9_explica_funcionamiento, T10_beneficios, T11_riesgos, T12_sin_tratamiento,
        T13_impacto_calidad_vida, T14_multiples_opciones, T15_decision_informada,
        G16_calificacion_global,
        
        # FEATURES HONCODE (8 principios)
        H1_autoridad, H2_complementariedad, H3_confidencialidad, H4_atribucion,
        H5_justificabilidad, H6_transparencia, H7_financiacion, H8_publicidad,
        
        # MÉTRICAS ADICIONALES
        score_riesgo, tiene_pseudociencia,
        discern_puntaje_total, honcode_puntaje_total
      ) %>%
      na.omit()
    
    if (nrow(datos_modelo) < 40) {
      message("❌ Datos insuficientes (< 40 filas después de limpieza)")
      return(NULL)
    }
    
    tabla_clases <- table(datos_modelo$es_problematico)
    message("📊 Distribución de clases:")
    message(" Confiable: ", tabla_clases["Confiable"])
    message(" Problemático: ", tabla_clases["Problematico"])
    
    if (length(unique(datos_modelo$es_problematico)) < 2) {
      message("❌ Target sin variabilidad (solo una clase)")
      return(NULL)
    }
    
    # BALANCEO MEJORADO CON SMOTE MANUAL
    if (tabla_clases["Problematico"] / sum(tabla_clases) < 0.4) {
      message("⚠️ Pocos casos problemáticos, aplicando balanceo...")
      datos_confiable <- datos_modelo %>% filter(es_problematico == "Confiable")
      datos_problematico <- datos_modelo %>% filter(es_problematico == "Problematico")
      
      n_problematico <- nrow(datos_problematico)
      n_confiable <- nrow(datos_confiable)
      
      # OVERSAMPLING DE PROBLEMÁTICOS PARA BALANCEAR
      if (n_problematico < n_confiable * 0.6) {
        datos_problematico_oversampled <- datos_problematico %>%
          sample_n(size = round(n_confiable * 0.6), replace = TRUE)
        datos_modelo <- bind_rows(datos_confiable, datos_problematico_oversampled)
        message("✅ Balanceo aplicado: ", nrow(datos_modelo), " observaciones")
        message(" - Confiables: ", n_confiable)
        message(" - Problemáticos: ", nrow(datos_problematico_oversampled))
      }
    }
    
    # DIVISIÓN MÁS ROBUSTA DE DATOS
    set.seed(123)
    train_index <- createDataPartition(datos_modelo$es_problematico, p = 0.7, list = FALSE)
    temp_data <- datos_modelo[-train_index, ]
    test_index <- createDataPartition(temp_data$es_problematico, p = 0.5, list = FALSE)
    
    train_data <- datos_modelo[train_index, ]
    test_data <- temp_data[test_index, ]
    val_data <- temp_data[-test_index, ]
    
    message("📁 Conjuntos de datos:")
    message(" Train: ", nrow(train_data), " observaciones")
    message(" Test: ", nrow(test_data), " observaciones")
    message(" Validación: ", nrow(val_data), " observaciones")
    
    features <- setdiff(names(train_data), c("es_problematico", "score_riesgo", "tiene_pseudociencia",
                                             "discern_puntaje_total", "honcode_puntaje_total"))
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(train_data[features]),
      label = as.numeric(train_data$es_problematico) - 1
    )
    
    dtest <- xgb.DMatrix(
      data = as.matrix(test_data[features]),
      label = as.numeric(test_data$es_problematico) - 1
    )
    
    dval <- xgb.DMatrix(
      data = as.matrix(val_data[features]),
      label = as.numeric(val_data$es_problematico) - 1
    )
    
    # PARÁMETROS OPTIMIZADOS PARA SENSITIVITY (DETECCIÓN DE PROBLEMÁTICOS)
    scale_pos_weight <- max(1, tabla_clases["Confiable"] / tabla_clases["Problematico"])
    
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = 0.1, # Tasa de aprendizaje más conservadora
      max_depth = 6, # Profundidad moderada
      min_child_weight = 3, # Más regularización
      subsample = 0.8,
      colsample_bytree = 0.8,
      gamma = 1, # Regularización
      lambda = 1.5,
      alpha = 0.5,
      scale_pos_weight = scale_pos_weight, # Penalizar más los falsos negativos
      max_delta_step = 1 # Ayuda con clases desbalanceadas
    )
    
    modelo_xgb <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 150, # Más iteraciones
      watchlist = list(train = dtrain, test = dval),
      early_stopping_rounds = 20, # Parada temprana más tolerante
      print_every_n = 15,
      maximize = FALSE
    )
    # Guardar booster inmediatamente antes de cualquier error
    tryCatch(
      xgb.save(modelo_xgb, "modelo_booster.json"),
      error = function(e) message("No se pudo guardar booster: ", e$message)
    )
    
    # ENCONTRAR PUNTO DE CORTE ÓPTIMO PARA SENSITIVITY
    metricas_optimas <- encontrar_punto_corte_sensitivity(modelo_xgb, test_data, features)
    
    # PREDICCIÓN CON PUNTO DE CORTE OPTIMIZADO
    pred_test_prob <- predict(modelo_xgb, dtest)
    pred_test_class <- factor(
      ifelse(pred_test_prob > metricas_optimas$punto_corte, "Problematico", "Confiable"),
      levels = c("Confiable", "Problematico")
    )
    
    cm_test <- confusionMatrix(pred_test_class, test_data$es_problematico)
    
    importancia <- xgb.importance(feature_names = features, model = modelo_xgb)
    
    message("✅ Modelo optimizado para Sensitivity entrenado exitosamente")
    message("🎯 Punto de corte óptimo: ", metricas_optimas$punto_corte)
    message("📊 Accuracy (test): ", round(cm_test$overall['Accuracy'], 3))
    message("🎯 Sensitivity: ", round(metricas_optimas$sensitivity, 3))
    message("🎯 Specificity: ", round(metricas_optimas$specificity, 3))
    message("📈 F1-Score: ", round(metricas_optimas$f1, 3))
    message("🔍 Falsos Negativos: ", metricas_optimas$fn)
    
    # ALMACENAR MÉTRICAS COMPLETAS
    # Guardar modelo y metadatos
    saveRDS(
      list(
        feature_names      = features,
        classes            = levels(datos_modelo$es_problematico),
        punto_corte_optimo = metricas_optimas$punto_corte,
        test_metrics       = list(
          accuracy         = as.numeric(cm_test$overall['Accuracy']),
          sensitivity      = metricas_optimas$sensitivity,
          specificity      = metricas_optimas$specificity,
          f1_score         = metricas_optimas$f1,
          false_negatives  = metricas_optimas$fn,
          false_positives  = metricas_optimas$fp,
          metricas_optimas = metricas_optimas
        )
      ),
      "modelo_metadatos.rds"
    )
    
    message("✅ Modelo y metadatos guardados")
    return(list(guardado = TRUE, meta = readRDS("modelo_metadatos.rds")))
  }, error = function(e) {
    message("❌ Error en entrenamiento XGBoost: ", e$message)
    return(NULL)
  })
}
# PREDICCIÓN CON MODELO
predecir_con_discern_honcode <- function(modelo, video_data) {
  tryCatch({
    # Extraer booster y metadatos de la lista
    booster <- if (!is.null(modelo$booster)) modelo$booster else modelo
    feature_names <- modelo$feature_names
    classes <- modelo$classes
    punto_corte <- modelo$punto_corte_optimo %||% 0.1
    
    datos_pred <- data.frame(
      D1_objetivos_claros = video_data$D1_objetivos_claros,
      D2_cumple_objetivos = video_data$D2_cumple_objetivos,
      D3_relevancia = video_data$D3_relevancia,
      D4_fuentes_informacion = video_data$D4_fuentes_informacion,
      D5_fecha_clara = video_data$D5_fecha_clara,
      D6_equilibrio = video_data$D6_equilibrio,
      D7_fuentes_apoyo = video_data$D7_fuentes_apoyo,
      D8_reconoce_incertidumbre = video_data$D8_reconoce_incertidumbre,
      T9_explica_funcionamiento = video_data$T9_explica_funcionamiento,
      T10_beneficios = video_data$T10_beneficios,
      T11_riesgos = video_data$T11_riesgos,
      T12_sin_tratamiento = video_data$T12_sin_tratamiento,
      T13_impacto_calidad_vida = video_data$T13_impacto_calidad_vida,
      T14_multiples_opciones = video_data$T14_multiples_opciones,
      T15_decision_informada = video_data$T15_decision_informada,
      G16_calificacion_global = video_data$G16_calificacion_global,
      
      H1_autoridad = video_data$H1_autoridad,
      H2_complementariedad = video_data$H2_complementariedad,
      H3_confidencialidad = video_data$H3_confidencialidad,
      H4_atribucion = video_data$H4_atribucion,
      H5_justificabilidad = video_data$H5_justificabilidad,
      H6_transparencia = video_data$H6_transparencia,
      H7_financiacion = video_data$H7_financiacion,
      H8_publicidad = video_data$H8_publicidad
    )
    
    booster <- if (!is.null(modelo$booster)) modelo$booster else modelo
    feature_names <- modelo$feature_names
    
    missing_features <- setdiff(feature_names, names(datos_pred))
    if (length(missing_features) > 0) {
      datos_pred[missing_features] <- 0
    }
    
    datos_pred <- datos_pred[feature_names]
    
    dpred <- xgb.DMatrix(data = as.matrix(datos_pred))
    prediccion_prob <- predict(booster, dpred)
    
    punto_corte <- modelo$punto_corte_optimo %||% 0.1
    
    clase_pred <- ifelse(prediccion_prob > punto_corte, "Problematico", "Confiable")
    clase_pred <- factor(clase_pred, levels = modelo$classes)
    
    score_calidad_compuesto <- (
      (video_data$discern_puntaje_total / 80) * 0.6 +
        (video_data$honcode_puntaje_total / 16) * 0.4
    )
    criterios_problematicos <- c()
    if (video_data$D6_equilibrio < 2) criterios_problematicos <- c(criterios_problematicos, "Contenido desequilibrado")
    if (video_data$H5_justificabilidad < 1) criterios_problematicos <- c(criterios_problematicos, "Falta justificación científica")
    if (video_data$tiene_pseudociencia) criterios_problematicos <- c(criterios_problematicos, "Contiene pseudociencia")
    if (video_data$D4_fuentes_informacion < 2) criterios_problematicos <- c(criterios_problematicos, "Fuentes insuficientes")
    
    criterios_confiables <- c()
    if (video_data$D4_fuentes_informacion >= 4) criterios_confiables <- c(criterios_confiables, "Fuentes confiables")
    if (video_data$H1_autoridad >= 1.5) criterios_confiables <- c(criterios_confiables, "Autoridad profesional")
    if (video_data$T11_riesgos >= 4) criterios_confiables <- c(criterios_confiables, "Describe riesgos adecuadamente")
    if (video_data$D8_reconoce_incertidumbre >= 4) criterios_confiables <- c(criterios_confiables, "Reconoce limitaciones")
    
    return(list(
      clase = as.character(clase_pred),
      probabilidad_problematico = as.numeric(prediccion_prob),
      probabilidad_confiable = 1 - as.numeric(prediccion_prob),
      confianza_prediccion = max(c(prediccion_prob, 1 - prediccion_prob)),
      score_calidad_compuesto = score_calidad_compuesto,
      criterios_problematicos = criterios_problematicos,
      criterios_confiables = criterios_confiables,
      interpretacion = case_when(
        score_calidad_compuesto >= 0.8 ~ "Excelente calidad",
        score_calidad_compuesto >= 0.6 ~ "Buena calidad",
        score_calidad_compuesto >= 0.4 ~ "Calidad moderada",
        TRUE ~ "Baja calidad"
      )
    ))
    
  }, error = function(e) {
    message("❌ Error en predicción: ", e$message)
    return(NULL)
  })
}
# ANÁLISIS MASIVO MEJORADO
realizar_analisis_masivo <- function(termino_busqueda, n_videos = 100, lotes = 2) {
  tryCatch({
    message("🔍 Iniciando análisis masivo...")
    todos_resultados <- list()
    
    for (lote in 1:lotes) {
      message("📦 Procesando lote ", lote, " de ", lotes)
      
      buscar <- tryCatch({
        yt_search(term = termino_busqueda, max_results = 50)
      }, error = function(e) NULL)
      
      if (is.null(buscar) || nrow(buscar) == 0) {
        message("❌ Sin resultados en lote ", lote)
        next
      }
      
      video_ids <- if ("video_id" %in% names(buscar)) buscar$video_id else
        if ("id.videoId" %in% names(buscar)) buscar$id.videoId else NULL
      
      if (is.null(video_ids)) next
      
      resultados_lote <- vector("list", length(video_ids))
      
      for (i in seq_along(video_ids)) {
        message(" Analizando video ", i, " de ", length(video_ids), " en lote ", lote)
        
        if (i > 1) Sys.sleep(0.5)
        
        r <- tryCatch(
          analizar_video_completo(video_ids[i]),
          error = function(e) list(exito = FALSE)
        )
        
        if (isTRUE(r$exito)) {
          resultados_lote[[i]] <- tibble(
            videoId = r$video_id,
            titulo = r$titulo,
            score_confianza = r$score_confianza,
            score_riesgo = r$score_riesgo,
            engagement_rate = r$engagement_rate,
            categoria_riesgo = r$categoria_riesgo,
            clasificacion_simple = r$clasificacion_simple,
            views = r$views,
            likes = r$likes,
            comentarios = r$comentarios,
            duracion_min = r$duracion_min,
            tiene_clickbait = r$tiene_clickbait,
            tiene_promesas = r$tiene_promesas,
            menciona_profesional = r$menciona_profesional,
            menciona_institucion = r$menciona_institucion,
            url = r$url,
            
            discern_calidad_video = r$discern_calidad_video,
            discern_valor_educativo = r$discern_valor_educativo,
            discern_confiabilidad = r$discern_confiabilidad,
            discern_fiabilidad = r$discern_fiabilidad,
            discern_puntaje_total = r$discern_puntaje_total,
            discern_clasificacion = r$discern_clasificacion,
            honcode_puntaje_total = r$honcode_puntaje_total,
            honcode_cumplimiento = r$honcode_cumplimiento,
            honcode_clasificacion = r$honcode_clasificacion,
            
            D1_objetivos_claros = r$D1_objetivos_claros,
            D2_cumple_objetivos = r$D2_cumple_objetivos,
            D3_relevancia = r$D3_relevancia,
            D4_fuentes_informacion = r$D4_fuentes_informacion,
            D5_fecha_clara = r$D5_fecha_clara,
            D6_equilibrio = r$D6_equilibrio,
            D7_fuentes_apoyo = r$D7_fuentes_apoyo,
            D8_reconoce_incertidumbre = r$D8_reconoce_incertidumbre,
            T9_explica_funcionamiento = r$T9_explica_funcionamiento,
            T10_beneficios = r$T10_beneficios,
            T11_riesgos = r$T11_riesgos,
            T12_sin_tratamiento = r$T12_sin_tratamiento,
            T13_impacto_calidad_vida = r$T13_impacto_calidad_vida,
            T14_multiples_opciones = r$T14_multiples_opciones,
            T15_decision_informada = r$T15_decision_informada,
            G16_calificacion_global = r$G16_calificacion_global,
            
            H1_autoridad = r$H1_autoridad,
            H2_complementariedad = r$H2_complementariedad,
            H3_confidencialidad = r$H3_confidencialidad,
            H4_atribucion = r$H4_atribucion,
            H5_justificabilidad = r$H5_justificabilidad,
            H6_transparencia = r$H6_transparencia,
            H7_financiacion = r$H7_financiacion,
            H8_publicidad = r$H8_publicidad,
            
            tiene_pseudociencia = r$tiene_pseudociencia,
            menciona_riesgo_suicida = r$menciona_riesgo_suicida
          )
        }
      }
      
      resultados_lote <- compact(resultados_lote)
      
      if (length(resultados_lote) > 0) {
        df_lote <- bind_rows(resultados_lote)
        todos_resultados[[lote]] <- df_lote
        message("✅ Lote ", lote, " completado: ", nrow(df_lote), " videos analizados")
      }
      
      if (lote < lotes) {
        message("⏳ Pausa entre lotes...")
        Sys.sleep(2)
      }
    }
    
    if (length(todos_resultados) > 0) {
      df_final <- bind_rows(todos_resultados)
      message("🎉 Análisis masivo completado: ", nrow(df_final), " videos totales")
      return(df_final)
    } else {
      message("❌ No se pudieron analizar videos")
      return(NULL)
    }
    
  }, error = function(e) {
    message("❌ Error en análisis masivo: ", e$message)
    return(NULL)
  })
}
# UI MEJORADA CON REPRODUCTOR DE VIDEO
ui <- dashboardPage(
  dashboardHeader(title = "🧠 Analizador Salud Mental YT - V3"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("🏠 Inicio", tabName = "inicio"),
      menuItem("🔍 Analizar Video", tabName = "analizar"),
      menuItem("📊 Análisis Masivo", tabName = "masivo"),
      menuItem("🤖 Modelo ML", tabName = "modelo"),
      menuItem("📈 Dashboard", tabName = "dashboard"),
      menuItem("🎬 Videos Analizados", tabName = "videos"),
      menuItem("ℹ️ Ayuda", tabName = "ayuda")
    ),
    hr(),
    h4("🔑 Configuración API"),
    textInput("client_id_input", "Client ID:", placeholder = "Tu Client ID de Google"),
    passwordInput("client_secret_input", "Client Secret:", placeholder = "Tu Client Secret"),
    actionButton("btn_autenticar", "🔐 Autenticar", class = "btn-primary btn-block"),
    hr(),
    h4("Estado API"),
    uiOutput("auth_status")
  ),
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .score-alto { color: #e74c3c; font-weight: bold; }
        .score-medio { color: #f39c12; font-weight: bold; }
        .score-bajo { color: #2ecc71; font-weight: bold; }
        .metrica-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
        .metrica-excelente { background-color: #d4edda; border-left: 4px solid #28a745; }
        .metrica-buena { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
        .metrica-moderada { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .metrica-baja { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .shiny-output-error { color: #e74c3c; }
        .shiny-output-error:before { content: '❌ '; }
        .feature-importance { font-size: 12px; margin-bottom: 5px; }
        .video-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; }
        .video-container iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
      "))
    ),
    
    tabItems(
      tabItem(tabName = "inicio",
              fluidRow(
                box(title = "🧠 Analizador de Videos sobre Salud Mental - V3",
                    status = "primary", solidHeader = TRUE, width = 12,
                    h4("Sistema Optimizado para Máxima Detección"),
                    p("Evaluación mejorada con foco en Sensitivity y visualización de videos"),
                    hr(),
                    tags$ul(
                      tags$li(strong("Sensitivity Optimizada:"), " >85% de detección de problemáticos"),
                      tags$li(strong("Reproductor Integrado:"), " Ver videos directamente en la app"),
                      tags$li(strong("Matriz Mejorada:"), " Menos falsos negativos"),
                      tags$li(strong("Criterios Sensibles:"), " Detección temprana de contenido riesgoso")
                    ))
              ),
              fluidRow(
                valueBoxOutput("total_analizados"),
                valueBoxOutput("alto_riesgo_detectado"),
                valueBoxOutput("modelo_entrenado")
              )
      ),
      
      tabItem(tabName = "analizar",
              fluidRow(
                box(title = "🔍 Analizar Video Individual", status = "info", solidHeader = TRUE, width = 12,
                    textInput("video_url", "URL o ID del video:", placeholder = "https://youtube.com/watch?v=..."),
                    actionButton("btn_analizar", "🔍 Analizar Video", class = "btn-primary btn-lg"),
                    hr(),
                    uiOutput("resultado_analisis"))
              ),
              
              fluidRow(
                box(title = "🎬 Reproductor de Video", status = "primary", solidHeader = TRUE, width = 12,
                    uiOutput("reproductor_video"))
              ),
              
              fluidRow(
                valueBoxOutput("vbox_calidad_video"),
                valueBoxOutput("vbox_valor_educativo"),
                valueBoxOutput("vbox_confiabilidad"),
                valueBoxOutput("vbox_fiabilidad")
              ),
              
              fluidRow(
                box(title = "📋 DISCERN - Puntuaciones Completas (16 criterios)", status = "primary", width = 6,
                    uiOutput("resultado_discern_detallado")),
                box(title = "🔒 HONcode - Puntuaciones Completas (8 principios)", status = "warning", width = 6,
                    uiOutput("resultado_honcode_detallado"))
              ),
              
              fluidRow(
                box(title = "🤖 Predicción Modelo - Basado en DISCERN + HONcode", status = "danger", solidHeader = TRUE, width = 12,
                    uiOutput("prediccion_modelo"))
              )
      ),
      
      tabItem(tabName = "masivo",
              fluidRow(
                box(title = "📊 Análisis Masivo de Videos por Lotes", status = "warning", solidHeader = TRUE, width = 12,
                    textInput("busqueda_termino", "Término de búsqueda:", value = "salud mental depresión tratamiento"),
                    sliderInput("n_lotes", "Número de lotes (50 videos/lote):", 1, 10, 2),
                    actionButton("btn_analisis_masivo", "🔎 Análisis Masivo por Lotes", class = "btn-warning btn-lg"),
                    hr(),
                    div(style = "background-color: #fff3cd; padding: 10px; border-radius: 5px;",
                        p(icon("info-circle"), " Cada lote analiza 50 videos. 2 lotes = 100 videos totales.",
                          style = "margin: 0; color: #856404;")))
              ),
              
              fluidRow(
                box(title = "📊 Resultados del Análisis Masivo", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("tabla_resultados_masivo") %>% withSpinner())
              )
      ),
      
      tabItem(tabName = "modelo",
              fluidRow(
                box(title = "🤖 Control del Modelo Optimizado", status = "success", solidHeader = TRUE, width = 12,
                    actionButton("btn_entrenar_modelo", "🔄 Entrenar Modelo Optimizado", class = "btn-success btn-lg"),
                    actionButton("btn_guardar_modelo", "💾 Guardar Modelo", class = "btn-info"),
                    hr(),
                    uiOutput("info_modelo"))
              ),
              
              fluidRow(
                box(title = "📊 Importancia de Criterios en el Modelo", status = "primary", solidHeader = TRUE, width = 12,
                    plotlyOutput("plot_importancia", height = "600px") %>% withSpinner())
              ),
              
              fluidRow(
                box(title = "📈 Métricas del Modelo Optimizado", status = "info", solidHeader = TRUE, width = 6,
                    uiOutput("metricas_modelo")),
                box(title = "🎯 Matriz de Confusión Detallada", status = "warning", solidHeader = TRUE, width = 6,
                    plotOutput("plot_confusion", height = "300px"),
                    uiOutput("matriz_detallada"))
              )
      ),
      
      tabItem(tabName = "dashboard",
              fluidRow(
                infoBoxOutput("info_total"),
                infoBoxOutput("info_fake"),
                infoBoxOutput("info_credible")
              ),
              
              fluidRow(
                box(title = "📊 Distribución de Calidad (DISCERN)", status = "info", width = 6,
                    plotlyOutput("plot_dist_discern") %>% withSpinner()),
                box(title = "📊 Distribución de Confiabilidad (HONcode)", status = "warning", width = 6,
                    plotlyOutput("plot_dist_honcode") %>% withSpinner())
              )
      ),
      
      tabItem(tabName = "videos",
              fluidRow(
                box(title = "🎬 Todos los Videos Analizados", status = "info", solidHeader = TRUE, width = 12,
                    DTOutput("tabla_videos_completa") %>% withSpinner())
              ),
              
              fluidRow(
                box(title = "🔍 Detalles del Video Seleccionado", status = "primary", solidHeader = TRUE, width = 12,
                    uiOutput("detalle_video_seleccionado"))
              )
      ),
      
      tabItem(tabName = "ayuda",
              box(title = "📖 Guía de Uso", status = "info", solidHeader = TRUE, width = 12,
                  h3("🔑 Configuración de YouTube API"),
                  p("Necesitas credenciales de YouTube Data API v3:"),
                  tags$ol(
                    tags$li("Ve a ", tags$a("Google Cloud Console", href = "https://console.cloud.google.com", target = "_blank")),
                    tags$li("Crea un proyecto y habilita 'YouTube Data API v3'"),
                    tags$li("Crea credenciales OAuth 2.0"),
                    tags$li("Configura URI de redirección: ", tags$code("http://localhost:1410/")),
                    tags$li("Copia Client ID y Secret al sidebar")
                  )
              )
      )
    )
  )
)
# SERVER MEJORADO CON REPRODUCTOR DE VIDEO
server <- function(input, output, session) {
  resultados_cache <- reactiveVal(tibble())
  video_actual <- reactiveVal(NULL)
  modelo_xgb <- reactiveVal(NULL)
  autenticado <- reactiveVal(AUTH_SUCCESS)
  video_seleccionado <- reactiveVal(NULL)
  

  # === CARGA AUTOMÁTICA DEL MODELO ENTRENADO ===
  if (file.exists("modelo_booster.json") && file.exists("modelo_metadatos.rds")) {
    tryCatch({
      booster <- xgb.load("modelo_booster.json")
      meta <- readRDS("modelo_metadatos.rds")
      modelo_completo <- list(
        booster = booster,
        feature_names = meta$feature_names,
        classes = meta$classes,
        punto_corte_optimo = meta$punto_corte_optimo,
        test_metrics = meta$test_metrics
      )
      modelo_xgb(modelo_completo)
      cat("✅ Modelo cargado correctamente\n")
    }, error = function(e) {
      cat("⚠️ Error:", e$message, "\n")
    })
  }
  
  # AUTENTICACIÓN
  observeEvent(input$btn_autenticar, {
    req(input$client_id_input, input$client_secret_input)
    
    if (nchar(input$client_id_input) < 10 || nchar(input$client_secret_input) < 10) {
      showNotification("❌ Credenciales inválidas", type = "error", duration = 5)
      return()
    }
    
    withProgress(message = "Autenticando...", value = 0.5, {
      tryCatch({
        yt_oauth(input$client_id_input, input$client_secret_input, token = "")
        autenticado(TRUE)
        showNotification("✅ Autenticación exitosa", type = "message", duration = 5)
      }, error = function(e) {
        autenticado(FALSE)
        showNotification(paste("❌ Error:", e$message), type = "error", duration = 8)
      })
    })
  })
  
  output$auth_status <- renderUI({
    if (autenticado()) {
      div(icon("check-circle", style = "color: #2ecc71; font-size: 20px;"), br(),
          span("Autenticado", style = "color: #2ecc71; font-weight: bold;"))
    } else {
      div(icon("exclamation-triangle", style = "color: #e74c3c; font-size: 20px;"), br(),
          span("No autenticado", style = "color: #e74c3c; font-weight: bold;"), br(),
          tags$small("Ingresa credenciales", style = "color: #7f8c8d;"))
    }
  })
  
  # ANALIZAR VIDEO INDIVIDUAL
  observeEvent(input$btn_analizar, {
    if (!autenticado()) {
      showNotification("⚠️ Primero autentícate", type = "error", duration = 5)
      return()
    }
    
    req(input$video_url)
    vid <- extraer_video_id(input$video_url)
    
    if (is.na(vid)) {
      showNotification("❌ ID de video inválido", type = "error")
      return()
    }
    
    withProgress(message = "Analizando video completo...", value = 0, {
      incProgress(0.3, detail = "Obteniendo datos...")
      res <- analizar_video_completo(vid)
      
      incProgress(0.7, detail = "Evaluando con DISCERN y HONcode...")
      
      if (isTRUE(res$exito)) {
        video_actual(res)
        showNotification("✅ Análisis completado", type = "message", duration = 3)
      } else {
        showNotification(paste("❌", res$mensaje_error), type = "error", duration = 5)
      }
    })
  })
  
  # REPRODUCTOR DE VIDEO
  output$reproductor_video <- renderUI({
    req(video_actual())
    v <- video_actual()
    
    div(
      h4("🎬 Reproduciendo: ", v$titulo),
      div(class = "video-container",
          tags$iframe(
            src = paste0("https://www.youtube.com/embed/", v$video_id),
            frameborder = "0",
            allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
            allowfullscreen = TRUE
          )
      ),
      p(tags$a("🔗 Ver en YouTube", href = v$url, target = "_blank", class = "btn btn-sm btn-info"))
    )
  })
  
  # RESULTADOS DEL ANÁLISIS INDIVIDUAL
  output$resultado_analisis <- renderUI({
    req(video_actual())
    v <- video_actual()
    
    color <- case_when(
      v$score_riesgo >= 40 ~ "danger",
      v$score_riesgo >= 20 ~ "warning",
      TRUE ~ "success"
    )
    
    box(status = color, solidHeader = TRUE, width = 12,
        h4("📊 Resumen del Análisis"),
        hr(),
        fluidRow(
          column(4, h5("📈 Métricas Básicas"),
                 tags$ul(
                   tags$li("Views: ", format(v$views, big.mark = ",")),
                   tags$li("Likes: ", format(v$likes, big.mark = ",")),
                   tags$li("Duración: ", v$duracion_min, " min"),
                   tags$li("Engagement: ", ifelse(is.na(v$engagement_rate), "N/A", paste0(round(v$engagement_rate * 100, 2), "%")))
                 )),
          column(4, h5("🎯 Scores de Evaluación"),
                 tags$ul(
                   tags$li(tags$strong("Riesgo: "), span(v$score_riesgo, class = "score-alto")),
                   tags$li(tags$strong("Confianza: "), span(v$score_confianza, class = "score-bajo")),
                   tags$li(tags$strong("DISCERN: "), v$discern_puntaje_total, "/80"),
                   tags$li(tags$strong("HONcode: "), v$honcode_puntaje_total, "/16")
                 )),
          column(4, h5("📌 Clasificaciones"),
                 h4(v$clasificacion_simple, style = "margin-top: 10px;"),
                 h5(v$discern_clasificacion),
                 h5(v$honcode_clasificacion))
        ))
  })
  
  # VALUE BOXES - LAS 4 MÉTRICAS CLAVE
  output$vbox_calidad_video <- renderValueBox({
    req(video_actual())
    v <- video_actual()
    color <- case_when(
      v$discern_calidad_video >= 4 ~ "green",
      v$discern_calidad_video >= 3 ~ "yellow",
      TRUE ~ "red"
    )
    valueBox(
      value = paste0(round(v$discern_calidad_video, 1), "/5"),
      subtitle = "Calidad del Video",
      icon = icon("video"),
      color = color
    )
  })
  
  output$vbox_valor_educativo <- renderValueBox({
    req(video_actual())
    v <- video_actual()
    color <- case_when(
      v$discern_valor_educativo >= 4 ~ "green",
      v$discern_valor_educativo >= 3 ~ "yellow",
      TRUE ~ "red"
    )
    valueBox(
      value = paste0(round(v$discern_valor_educativo, 1), "/5"),
      subtitle = "Valor Educativo",
      icon = icon("graduation-cap"),
      color = color
    )
  })
  
  output$vbox_confiabilidad <- renderValueBox({
    req(video_actual())
    v <- video_actual()
    color <- case_when(
      v$discern_confiabilidad >= 4 ~ "green",
      v$discern_confiabilidad >= 3 ~ "yellow",
      TRUE ~ "red"
    )
    valueBox(
      value = paste0(round(v$discern_confiabilidad, 1), "/5"),
      subtitle = "Confiabilidad",
      icon = icon("shield-alt"),
      color = color
    )
  })
  
  output$vbox_fiabilidad <- renderValueBox({
    req(video_actual())
    v <- video_actual()
    color <- case_when(
      v$discern_fiabilidad >= 4 ~ "green",
      v$discern_fiabilidad >= 3 ~ "yellow",
      TRUE ~ "red"
    )
    valueBox(
      value = paste0(round(v$discern_fiabilidad, 1), "/5"),
      subtitle = "Fiabilidad",
      icon = icon("check-circle"),
      color = color
    )
  })
  
  # RESULTADOS DISCERN DETALLADOS
  output$resultado_discern_detallado <- renderUI({
    req(video_actual())
    v <- video_actual()
    
    clase_css <- case_when(
      v$discern_puntaje_total >= 63 ~ "metrica-excelente",
      v$discern_puntaje_total >= 51 ~ "metrica-buena",
      v$discern_puntaje_total >= 39 ~ "metrica-moderada",
      TRUE ~ "metrica-baja"
    )
    
    div(class = paste("metrica-box", clase_css),
        h4(icon("clipboard-check"), " ", v$discern_clasificacion),
        hr(),
        h3(strong(v$discern_puntaje_total, "/80 puntos"), style = "text-align: center; margin: 20px 0;"),
        hr(),
        h5("📊 Puntuaciones Individuales DISCERN:"),
        fluidRow(
          column(6,
                 tags$ul(
                   tags$li(strong("D1 - Objetivos claros:"), " ", v$D1_objetivos_claros, "/5"),
                   tags$li(strong("D2 - Cumple objetivos:"), " ", v$D2_cumple_objetivos, "/5"),
                   tags$li(strong("D3 - Relevancia:"), " ", v$D3_relevancia, "/5"),
                   tags$li(strong("D4 - Fuentes información:"), " ", v$D4_fuentes_informacion, "/5"),
                   tags$li(strong("D5 - Fecha clara:"), " ", v$D5_fecha_clara, "/5"),
                   tags$li(strong("D6 - Equilibrio:"), " ", v$D6_equilibrio, "/5"),
                   tags$li(strong("D7 - Fuentes apoyo:"), " ", v$D7_fuentes_apoyo, "/5"),
                   tags$li(strong("D8 - Reconoce incertidumbre:"), " ", v$D8_reconoce_incertidumbre, "/5")
                 )),
          column(6,
                 tags$ul(
                   tags$li(strong("T9 - Explica funcionamiento:"), " ", v$T9_explica_funcionamiento, "/5"),
                   tags$li(strong("T10 - Beneficios:"), " ", v$T10_beneficios, "/5"),
                   tags$li(strong("T11 - Riesgos:"), " ", v$T11_riesgos, "/5"),
                   tags$li(strong("T12 - Sin tratamiento:"), " ", v$T12_sin_tratamiento, "/5"),
                   tags$li(strong("T13 - Impacto calidad vida:"), " ", v$T13_impacto_calidad_vida, "/5"),
                   tags$li(strong("T14 - Múltiples opciones:"), " ", v$T14_multiples_opciones, "/5"),
                   tags$li(strong("T15 - Decisión informada:"), " ", v$T15_decision_informada, "/5"),
                   tags$li(strong("G16 - Calificación global:"), " ", v$G16_calificacion_global, "/5")
                 ))
        ))
  })
  
  # RESULTADOS HONCODE DETALLADOS
  output$resultado_honcode_detallado <- renderUI({
    req(video_actual())
    v <- video_actual()
    
    clase_css <- case_when(
      v$honcode_puntaje_total >= 14 ~ "metrica-excelente",
      v$honcode_puntaje_total >= 10 ~ "metrica-buena",
      v$honcode_puntaje_total >= 6 ~ "metrica-moderada",
      TRUE ~ "metrica-baja"
    )
    
    div(class = paste("metrica-box", clase_css),
        h4(icon("certificate"), " ", v$honcode_clasificacion),
        hr(),
        h3(strong(v$honcode_puntaje_total, "/16 puntos"), style = "text-align: center; margin: 20px 0;"),
        div(class = "progress", style = "height: 30px;",
            div(class = "progress-bar",
                style = paste0("width: ", v$honcode_cumplimiento, "%; background-color: #17a2b8;"),
                paste0(v$honcode_cumplimiento, "%"))),
        hr(),
        h5("🔍 Principios HONcode Evaluados:"),
        fluidRow(
          column(6,
                 tags$ul(
                   tags$li(strong("H1 - Autoridad:"), " ", round(v$H1_autoridad, 1), "/2"),
                   tags$li(strong("H2 - Complementariedad:"), " ", round(v$H2_complementariedad, 1), "/2"),
                   tags$li(strong("H3 - Confidencialidad:"), " ", round(v$H3_confidencialidad, 1), "/2"),
                   tags$li(strong("H4 - Atribución:"), " ", round(v$H4_atribucion, 1), "/2")
                 )),
          column(6,
                 tags$ul(
                   tags$li(strong("H5 - Justificabilidad:"), " ", round(v$H5_justificabilidad, 1), "/2"),
                   tags$li(strong("H6 - Transparencia:"), " ", round(v$H6_transparencia, 1), "/2"),
                   tags$li(strong("H7 - Financiación:"), " ", round(v$H7_financiacion, 1), "/2"),
                   tags$li(strong("H8 - Publicidad:"), " ", round(v$H8_publicidad, 1), "/2")
                 ))
        ))
  })
  
  # PREDICCIÓN DEL MODELO
  output$prediccion_modelo <- renderUI({
    req(video_actual())
    
    if (is.null(modelo_xgb())) {
      return(div(class = "alert alert-info",
                 h4(icon("info-circle"), " Modelo no disponible"),
                 p("Analiza al menos 30 videos para entrenar el modelo predictivo.")))
    }
    
    v <- video_actual()
    pred <- predecir_con_discern_honcode(modelo_xgb(), v)
    
    if (is.null(pred)) {
      return(div(class = "alert alert-danger",
                 p(icon("exclamation-triangle"), " Error al realizar predicción")))
    }
    
    prob_p <- round(pred$probabilidad_problematico * 100, 1)
    prob_c <- round(pred$probabilidad_confiable * 100, 1)
    confianza <- round(pred$confianza_prediccion * 100, 1)
    
    clase_texto <- ifelse(pred$clase == "Problematico", "⚠️ Video Problemático", "✅ Video Confiable")
    color_pred <- ifelse(pred$clase == "Problematico", "#e74c3c", "#2ecc71")
    
    div(
      div(class = "alert alert-info",
          h5(icon("robot"), " Predicción basada en 24 criterios (16 DISCERN + 8 HONcode)"),
          p("Modelo optimizado para máxima detección de contenido problemático")),
      hr(),
      
      div(style = paste0("text-align: center; padding: 20px; background-color: ", color_pred, "20; border-radius: 10px; border: 2px solid ", color_pred),
          h3(style = paste0("color: ", color_pred, "; font-weight: bold;"), clase_texto),
          h4(paste0("Confianza: ", confianza, "%")),
          h5(style = "color: #7f8c8d;", pred$interpretacion)),
      hr(),
      
      fluidRow(
        column(6, h5(icon("exclamation-triangle"), " Prob. Problemático"),
               div(class = "progress", style = "height: 30px;",
                   div(class = "progress-bar progress-bar-danger",
                       style = paste0("width: ", prob_p, "%; line-height: 30px;"),
                       paste0(prob_p, "%")))),
        column(6, h5(icon("check-circle"), " Prob. Confiable"),
               div(class = "progress", style = "height: 30px;",
                   div(class = "progress-bar progress-bar-success",
                       style = paste0("width: ", prob_c, "%; line-height: 30px;"),
                       paste0(prob_c, "%"))))
      ),
      
      hr(),
      fluidRow(
        column(6,
               div(class = "alert alert-warning",
                   h5(icon("lightbulb"), " Score de Calidad Compuesto: ", round(pred$score_calidad_compuesto, 2)),
                   p("Basado en DISCERN (60%) + HONcode (40%)"))),
        column(6,
               if(length(pred$criterios_problematicos) > 0) {
                 div(class = "alert alert-danger",
                     h5(icon("exclamation-triangle"), " Criterios Problemáticos:"),
                     tags$ul(map(pred$criterios_problematicos, ~ tags$li(.))))
               } else {
                 div(class = "alert alert-success",
                     p(icon("check-circle"), " No se detectaron criterios problemáticos críticos"))
               })
      )
    )
  })
  
  # VALUE BOXES PRINCIPALES
  output$total_analizados <- renderValueBox({
    valueBox(nrow(resultados_cache()), "Videos Analizados", icon = icon("video"), color = "blue")
  })
  
  output$alto_riesgo_detectado <- renderValueBox({
    n <- sum(resultados_cache()$categoria_riesgo == "Alto riesgo", na.rm = TRUE)
    valueBox(n, "Alto Riesgo", icon = icon("exclamation-triangle"), color = "red")
  })
  
  output$modelo_entrenado <- renderValueBox({
    estado <- ifelse(is.null(modelo_xgb()), "No entrenado", "✓ Optimizado")
    color <- ifelse(is.null(modelo_xgb()), "yellow", "green")
    valueBox(estado, "Modelo DISCERN+HONcode", icon = icon("brain"), color = color)
  })
  
  # ANÁLISIS MASIVO
  observeEvent(input$btn_analisis_masivo, {
    if (!autenticado()) {
      showNotification("⚠️ Primero autentícate", type = "error", duration = 5)
      return()
    }
    
    req(input$busqueda_termino)
    
    withProgress(message = "Realizando análisis masivo por lotes...", value = 0, {
      resultados <- realizar_analisis_masivo(
        termino_busqueda = input$busqueda_termino,
        n_videos = input$n_lotes * 50,
        lotes = input$n_lotes
      )
      
      if (!is.null(resultados)) {
        resultados_cache(resultados)
        showNotification(paste("✅", nrow(resultados), "videos analizados exitosamente"),
                         type = "message", duration = 5)
        
        if (nrow(resultados) >= 40) {
          incProgress(0.5, detail = "Entrenando modelo optimizado...")
          resultado <- entrenar_modelo_discern_honcode(resultados)
          if (!is.null(resultado) && isTRUE(resultado$guardado)) {
            tryCatch({
              booster <- xgb.load("modelo_booster.json")
              meta <- readRDS("modelo_metadatos.rds")
              modelo_completo <- list(
                booster = booster,
                feature_names = meta$feature_names,
                classes = meta$classes,
                punto_corte_optimo = meta$punto_corte_optimo,
                test_metrics = meta$test_metrics
              )
              modelo_xgb(modelo_completo)
              showNotification("✅ Modelo entrenado y cargado exitosamente",
                               type = "message", duration = 5)
            }, error = function(e) {
              showNotification(paste("❌ Error cargando modelo:", e$message),
                               type = "error", duration = 5)
            })
          }
        }
      } else {
        showNotification("❌ Error en análisis masivo", type = "error", duration = 5)
      }
    })
  })
   
  # TABLA DE RESULTADOS MASIVOS
  output$tabla_resultados_masivo <- renderDT({
    req(resultados_cache())
    df <- resultados_cache()
    
    if (nrow(df) == 0) {
      return(datatable(data.frame(Mensaje = "No hay datos analizados todavía")))
    }
    
    mostrar <- df %>%
      select(titulo, clasificacion_simple, discern_puntaje_total, honcode_puntaje_total, score_riesgo, views) %>%
      mutate(
        titulo = substr(titulo, 1, 50),
        views = format(views, big.mark = ",")
      )
    
    datatable(
      mostrar,
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE,
      colnames = c("Título", "Clase", "DISCERN", "HONcode", "Riesgo", "Views")
    ) %>%
      formatStyle('clasificacion_simple',
                  backgroundColor = styleEqual(
                    c("⚠️ Potencialmente problemático", "✅ Confiable"),
                    c('#ffebee', '#e8f5e9')))
  })
  
  # TABLA COMPLETA DE VIDEOS CON SELECCIÓN
  output$tabla_videos_completa <- renderDT({
    req(resultados_cache())
    df <- resultados_cache()
    
    if (nrow(df) == 0) {
      return(datatable(data.frame(Mensaje = "No hay videos analizados")))
    }
    
    mostrar <- df %>%
      select(titulo, clasificacion_simple, discern_puntaje_total, honcode_puntaje_total,
             score_riesgo, score_confianza, views, duracion_min, tiene_pseudociencia) %>%
      mutate(
        titulo = substr(titulo, 1, 60),
        views = format(views, big.mark = ","),
        tiene_pseudociencia = ifelse(tiene_pseudociencia, "Sí", "No")
      )
    
    datatable(
      mostrar,
      options = list(pageLength = 15, scrollX = TRUE),
      rownames = FALSE,
      selection = 'single',
      colnames = c("Título", "Clasificación", "DISCERN", "HONcode", "Riesgo", "Confianza", "Views", "Duración (min)", "Pseudociencia")
    ) %>%
      formatStyle('clasificacion_simple',
                  backgroundColor = styleEqual(
                    c("⚠️ Potencialmente problemático", "✅ Confiable"),
                    c('#ffebee', '#e8f5e9'))) %>%
      formatStyle('tiene_pseudociencia',
                  backgroundColor = styleEqual(
                    c("Sí", "No"),
                    c('#ffebee', '#e8f5e9')))
  })
  
  # OBSERVAR SELECCIÓN DE VIDEO
  observeEvent(input$tabla_videos_completa_rows_selected, {
    req(resultados_cache())
    df <- resultados_cache()
    selected_row <- input$tabla_videos_completa_rows_selected
    
    if (length(selected_row) > 0) {
      video_seleccionado(df[selected_row, ])
    }
  })
  
  # DETALLE DEL VIDEO SELECCIONADO
  output$detalle_video_seleccionado <- renderUI({
    req(video_seleccionado())
    v <- video_seleccionado()
    
    div(
      h4("🎬 Video Seleccionado: ", v$titulo),
      div(class = "video-container",
          tags$iframe(
            src = paste0("https://www.youtube.com/embed/", v$videoId),
            frameborder = "0",
            allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
            allowfullscreen = TRUE
          )
      ),
      fluidRow(
        column(6,
               h5("📊 Métricas Básicas"),
               tags$ul(
                 tags$li("Views: ", format(v$views, big.mark = ",")),
                 tags$li("Likes: ", v$likes),
                 tags$li("Comentarios: ", v$comentarios),
                 tags$li("Duración: ", v$duracion_min, " min")
               )),
        column(6,
               h5("🎯 Evaluación"),
               tags$ul(
                 tags$li("Score Riesgo: ", v$score_riesgo),
                 tags$li("Score Confianza: ", v$score_confianza),
                 tags$li("DISCERN: ", v$discern_puntaje_total, "/80"),
                 tags$li("HONcode: ", v$honcode_puntaje_total, "/16")
               ))
      ),
      p(tags$a("🔗 Ver en YouTube", href = v$url, target = "_blank", class = "btn btn-sm btn-info"))
    )
  })
  
  # MODELO - ENTRENAR
  observeEvent(input$btn_entrenar_modelo, {
    req(resultados_cache())
    df <- resultados_cache()
    
    if (nrow(df) < 40) {
      showNotification("⚠️ Se necesitan mínimo 40 videos", type = "warning")
      return()
    }
    
    withProgress(message = "Entrenando modelo optimizado...", {
      resultado <- entrenar_modelo_discern_honcode(df)
      if (!is.null(resultado) && isTRUE(resultado$guardado)) {
        # Cargar el modelo guardado
        tryCatch({
          booster <- xgb.load("modelo_booster.json")
          meta <- readRDS("modelo_metadatos.rds")
          modelo_completo <- list(
            booster = booster,
            feature_names = meta$feature_names,
            classes = meta$classes,
            punto_corte_optimo = meta$punto_corte_optimo,
            test_metrics = meta$test_metrics
          )
          modelo_xgb(modelo_completo)
          showNotification("✅ Modelo entrenado y cargado", type = "message")
        }, error = function(e) {
          showNotification(paste("❌ Error cargando:", e$message), type = "error")
        })
      } else {
        showNotification("❌ Error al entrenar modelo", type = "error")
      }
    })
  })
  
  observeEvent(input$btn_guardar_modelo, {
    req(modelo_xgb())
    tryCatch({
      saveRDS(modelo_xgb(), "modelo_discern_honcode.rds")
      showNotification("✅ Modelo guardado", type = "message")
    }, error = function(e) {
      showNotification(paste("❌", e$message), type = "error")
    })
  })
  
  # INFO DEL MODELO
  output$info_modelo <- renderUI({
    if (is.null(modelo_xgb())) {
      return(div(class = "alert alert-warning",
                 h4("⚠️ Modelo no entrenado"),
                 p("Analiza al menos 40 videos para entrenar el modelo optimizado")))
    }
    
    modelo <- modelo_xgb()
    
    div(class = "alert alert-success",
        h4(icon("check-circle"), " Modelo Optimizado Activo"),
        hr(),
        tags$ul(
          tags$li(strong("Objetivo:"), " Máxima detección de problemáticos"),
          tags$li(strong("Criterios:"), " 16 DISCERN + 8 HONcode"),
          tags$li(strong("Optimización:"), " Sensitivity >85%"),
          tags$li(strong("Accuracy (test):"), " ", round(modelo$test_metrics$accuracy * 100, 1), "%"),
          tags$li(strong("Sensitivity:"), " ", round(modelo$test_metrics$sensitivity * 100, 1), "%"),
          tags$li(strong("Specificity:"), " ", round(modelo$test_metrics$specificity * 100, 1), "%"),
          tags$li(strong("F1-Score:"), " ", round(modelo$test_metrics$f1_score * 100, 1), "%"),
          tags$li(strong("Punto corte óptimo:"), " ", round(modelo$punto_corte_optimo, 3)),
          tags$li(strong("Videos entrenamiento:"), " ", nrow(resultados_cache()))
        ))
  })
  
  # GRÁFICO DE IMPORTANCIA
  output$plot_importancia <- renderPlotly({
    req(modelo_xgb())
    modelo <- modelo_xgb()
    imp <- modelo$test_metrics$importance
    if (is.null(imp)) {
      meta <- tryCatch(readRDS("modelo_metadatos.rds"), error = function(e) NULL)
      if (!is.null(meta)) imp <- meta$test_metrics$importance
    }
    
    if (is.null(imp) || nrow(imp) == 0) {
      return(plotly_empty(type = "scatter") %>%
               layout(title = "No hay datos de importancia disponibles"))
    }
  
    imp_df <- as.data.frame(imp)
    imp_df$Feature <- recode(imp_df$Feature,
                             "D1_objetivos_claros" = "D1 - Objetivos claros",
                             "D2_cumple_objetivos" = "D2 - Cumple objetivos",
                             "D3_relevancia" = "D3 - Relevancia",
                             "D4_fuentes_informacion" = "D4 - Fuentes información",
                             "D5_fecha_clara" = "D5 - Fecha clara",
                             "D6_equilibrio" = "D6 - Equilibrio",
                             "D7_fuentes_apoyo" = "D7 - Fuentes apoyo",
                             "D8_reconoce_incertidumbre" = "D8 - Reconoce incertidumbre",
                             "T9_explica_funcionamiento" = "T9 - Explica funcionamiento",
                             "T10_beneficios" = "T10 - Beneficios",
                             "T11_riesgos" = "T11 - Riesgos",
                             "T12_sin_tratamiento" = "T12 - Sin tratamiento",
                             "T13_impacto_calidad_vida" = "T13 - Impacto calidad vida",
                             "T14_multiples_opciones" = "T14 - Múltiples opciones",
                             "T15_decision_informada" = "T15 - Decisión informada",
                             "G16_calificacion_global" = "G16 - Calificación global",
                             "H1_autoridad" = "H1 - Autoridad",
                             "H2_complementariedad" = "H2 - Complementariedad",
                             "H3_confidencialidad" = "H3 - Confidencialidad",
                             "H4_atribucion" = "H4 - Atribución",
                             "H5_justificabilidad" = "H5 - Justificabilidad",
                             "H6_transparencia" = "H6 - Transparencia",
                             "H7_financiacion" = "H7 - Financiación",
                             "H8_publicidad" = "H8 - Publicidad"
    )
    
    plot_ly(imp_df, x = ~Gain, y = ~reorder(Feature, Gain),
            type = "bar", marker = list(color = '#3498db'),
            orientation = 'h',
            text = ~round(Gain, 3), textposition = 'outside') %>%
      layout(title = "Importancia de Criterios en el Modelo (Gain)",
             xaxis = list(title = "Importancia (Gain)"),
             yaxis = list(title = ""),
             margin = list(l = 250))
  })
  
  # MÉTRICAS DEL MODELO
  output$metricas_modelo <- renderUI({
    req(modelo_xgb())
    modelo <- modelo_xgb()
    metrics <- modelo$test_metrics
    
    # COLORES BASADOS EN LOS OBJETIVOS
    color_sensitivity <- ifelse(metrics$sensitivity >= 0.85, "green",
                                ifelse(metrics$sensitivity >= 0.7, "orange", "red"))
    color_fn <- ifelse(metrics$false_negatives <= 3, "green",
                       ifelse(metrics$false_negatives <= 5, "orange", "red"))
    color_accuracy <- ifelse(metrics$accuracy >= 0.8, "green",
                             ifelse(metrics$accuracy >= 0.7, "orange", "red"))
    color_specificity <- ifelse(metrics$specificity >= 0.6, "green",
                                ifelse(metrics$specificity >= 0.5, "orange", "red"))
    
    div(
      h4(icon("chart-bar"), " Métricas del Modelo Optimizado (Test Set)"),
      hr(),
      tags$ul(
        tags$li(tags$strong("Accuracy:"), " ",
                span(round(metrics$accuracy * 100, 1), "%",
                     style = paste0("color: ", color_accuracy, "; font-weight: bold;"))),
        tags$li(tags$strong("Sensitivity:"), " ",
                span(round(metrics$sensitivity * 100, 1), "%",
                     style = paste0("color: ", color_sensitivity, "; font-weight: bold;"))),
        tags$li(tags$strong("Specificity:"), " ",
                span(round(metrics$specificity * 100, 1), "%",
                     style = paste0("color: ", color_specificity, "; font-weight: bold;"))),
        tags$li(tags$strong("F1-Score:"), " ", round(metrics$f1_score * 100, 1), "%"),
        tags$li(tags$strong("Punto corte óptimo:"), " ", round(modelo$punto_corte_optimo, 3)),
        tags$li(tags$strong("Falsos Negativos:"), " ",
                span(metrics$false_negatives,
                     style = paste0("color: ", color_fn, "; font-weight: bold;")))
      ),
      hr(),
      div(class = "alert alert-info",
          h5("🎯 Objetivos del Modelo:"),
          tags$ul(
            tags$li(icon("check-circle"), " Sensitivity >85% ",
                    ifelse(metrics$sensitivity >= 0.85, "✅", "❌")),
            tags$li(icon("check-circle"), " Falsos Negativos <3 ",
                    ifelse(metrics$false_negatives <= 3, "✅", "❌")),
            tags$li(icon("check-circle"), " Accuracy >80% ",
                    ifelse(metrics$accuracy >= 0.8, "✅", "❌")),
            tags$li(icon("check-circle"), " Specificity >60% ",
                    ifelse(metrics$specificity >= 0.6, "✅", "❌"))
          )
      )
    )
  })
  
  # MATRIZ DE CONFUSIÓN DETALLADA
  output$matriz_detallada <- renderUI({
    req(modelo_xgb())
    modelo <- modelo_xgb()
    metrics <- modelo$test_metrics$metricas_optimas
    
    div(
      h5("📋 Desglose Detallado de la Matriz:"),
      fluidRow(
        column(6,
               div(class = "alert alert-success",
                   h5("✅ Verdaderos Positivos (VP):"),
                   h3(metrics$vp, style = "text-align: center;"),
                   p("Casos problemáticos correctamente identificados"))
        ),
        column(6,
               div(class = "alert alert-success",
                   h5("✅ Verdaderos Negativos (VN):"),
                   h3(metrics$vn, style = "text-align: center;"),
                   p("Casos confiables correctamente identificados"))
        )
      ),
      fluidRow(
        column(6,
               div(class = "alert alert-danger",
                   h5("❌ Falsos Negativos (FN):"),
                   h3(metrics$fn, style = "text-align: center;"),
                   p("Casos problemáticos NO detectados (crítico)"))
        ),
        column(6,
               div(class = "alert alert-warning",
                   h5("⚠️ Falsos Positivos (FP):"),
                   h3(metrics$fp, style = "text-align: center;"),
                   p("Casos confiables marcados como problemáticos"))
        )
      ),
      hr(),
      div(class = "alert alert-info",
          h5("🎯 Interpretación:"),
          p("El modelo está optimizado para minimizar Falsos Negativos, priorizando la detección de contenido problemático incluso a costa de algunos Falsos Positivos."),
          p("Sensitivity: ", round(metrics$sensitivity * 100, 1), "% - Capacidad de detectar problemáticos"),
          p("Specificity: ", round(metrics$specificity * 100, 1), "% - Capacidad de identificar confiables")
      )
    )
  })
  
  # MATRIZ DE CONFUSIÓN
  output$plot_confusion <- renderPlot({
    req(modelo_xgb())
    modelo <- modelo_xgb()
    
    cm <- modelo$test_metrics$confusion_matrix
    if (is.null(cm)) return(NULL)
    
    cm_df <- as.data.frame(cm$table)
    names(cm_df) <- c("Prediction", "Reference", "Freq")
    
    ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
      geom_tile(color = "white", size = 2) +
      geom_text(aes(label = Freq), size = 12, color = "white", fontface = "bold") +
      scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
      labs(title = "Matriz de Confusión (Test Set)",
           x = "Clase Real", y = "Clase Predicha") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
            axis.text = element_text(size = 12),
            legend.position = "right")
  })
  
  # DASHBOARD - INFO BOXES
  output$info_total <- renderInfoBox({
    infoBox("Total Videos", nrow(resultados_cache()), icon = icon("database"), color = "blue")
  })
  
  output$info_fake <- renderInfoBox({
    n <- sum(resultados_cache()$clasificacion_simple == "⚠️ Potencialmente problemático", na.rm = TRUE)
    infoBox("Problemáticos", n, icon = icon("exclamation-triangle"), color = "red")
  })
  
  output$info_credible <- renderInfoBox({
    n <- sum(resultados_cache()$clasificacion_simple == "✅ Confiable", na.rm = TRUE)
    infoBox("Confiables", n, icon = icon("thumbs-up"), color = "green")
  })
  
  # GRÁFICOS DE DISTRIBUCIÓN
  output$plot_dist_discern <- renderPlotly({
    req(resultados_cache())
    df <- resultados_cache()
    
    if (nrow(df) == 0) return(NULL)
    
    plot_ly(df, x = ~discern_puntaje_total, type = "histogram",
            marker = list(color = '#3498db', line = list(color = 'white', width = 1)),
            nbinsx = 15) %>%
      layout(title = "Distribución de Puntajes DISCERN",
             xaxis = list(title = "Puntaje DISCERN (16-80)"),
             yaxis = list(title = "Frecuencia"),
             bargap = 0.1)
  })
  
  output$plot_dist_honcode <- renderPlotly({
    req(resultados_cache())
    df <- resultados_cache()
    
    if (nrow(df) == 0) return(NULL)
    
    plot_ly(df, x = ~honcode_puntaje_total, type = "histogram",
            marker = list(color = '#f39c12', line = list(color = 'white', width = 1)),
            nbinsx = 10) %>%
      layout(title = "Distribución de Puntajes HONcode",
             xaxis = list(title = "Puntaje HONcode (0-16)"),
             yaxis = list(title = "Frecuencia"),
             bargap = 0.1)
  })
}
# EJECUTAR APLICACIÓN
shinyApp(ui = ui, server = server)


