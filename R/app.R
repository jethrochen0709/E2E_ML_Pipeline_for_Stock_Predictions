library(shiny); library(ggplot2); library(readr)

ui <- fluidPage(
  titlePanel("Stock Predictions"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model","Model:",
                  choices=c("Scikit-Learn"="sklearn","PyTorch"="pytorch")),
      selectInput("ticker","Ticker:",choices=NULL)
    ),
    mainPanel(plotOutput("plot"), verbatimTextOutput("accuracy"))
  )
)

server <- function(input, output, session) {
  # discover tickers from output folder
  root <- if(basename(getwd())=="R") ".." else "."
  sk_files <- list.files(file.path(root,"output"),
                         pattern="^predictions_sklearn_.*\\.csv$")
  tickers <- sub("^predictions_sklearn_(.*)\\.csv$","\\1",sk_files)
  updateSelectInput(session,"ticker",choices=sort(tickers))

  get_df <- reactive({
    req(input$ticker, input$model)
    path <- file.path(root,"output",
      sprintf("predictions_%s_%s.csv",input$model,input$ticker))
    if(!file.exists(path)) return(NULL)
    df <- read_csv(path,show_col_types=FALSE)
    date_col <- colnames(df)[1]
    parsed <- as.Date(df[[date_col]])
    df$Date <- if(all(is.na(parsed))) seq_len(nrow(df)) else parsed
    df$Prediction <- as.numeric(df$Prediction)
    df$Direction  <- as.numeric(df$Direction)
    df
  })

  output$plot <- renderPlot({
    df <- get_df(); req(df)
    ggplot(df,aes(Date))+
      geom_line(aes(y=Close),color="black")+
      geom_point(aes(y=Close, color=(Prediction==Direction)),size=2)+
      scale_color_manual(values=c("TRUE"="green","FALSE"="red"),
                         labels=c("Correct","Incorrect"),name="Pred")+
      ggtitle(paste(input$ticker,input$model))
  })

  output$accuracy <- renderPrint({
    df <- get_df(); req(df)
    cat("Accuracy:",
        round(mean(df$Prediction==df$Direction)*100,2),"%")
  })
}

shinyApp(ui, server)