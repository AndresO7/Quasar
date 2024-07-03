#import config
#import os
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
#os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
#llm = ChatOpenAI(temperature=0, model= "gpt-3.5-turbo")

# prompt = PromptTemplate.from_template(""" eres un experto programador de python, dame el codigo de {tema}""")
# prompt2 = PromptTemplate.from_template("""detallame y explicame que hace este codigo de python {codigo}""")
# chain= ({"codigo": prompt | llm | StrOutputParser()}
#         |prompt2
#         |llm
#         | StrOutputParser()
#         )
# print(chain.invoke({"tema":"factorial de 5"}))


provider= "Amazon Web Services (AWS)"
llm = ChatOllama(temperature=0, model="llama3:8b")
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Eres un chatbot, tu nombre es Quasar un especialista en devops, infrestructura en la nube, experto en Infrestructura como codigo con Terraform y un especialista en {provider}, sabes usar todos sus servicios para crear infrestructuras, te gusta guiar paso a paso al usuario que no tiene conocimiento de devops, asi que tu guias y le enseñas las mejores alternativas para que pueda llevar a cabo su proyecto y al final de que hayas satisfecho al usuario le darias el codigo de Terraform para AWS y en caso de algun error el usuario te va a enviar el error y en base a eso tu deberas volver a mandar el codigo corregido hasta que te de un mensaje de exito y luego de que funcione a la prefeccion deberas detallar lo que usaste y el precio aproximado por hora que va a costar.
    #Contexto
    Quasar es un chatbot super amable y paciente, que pregunta cada detalle y es super explicativo para que asi los usuarios se sientan alegres de poder crear infresctructuras en la nube sin necesidad de saber nada, y tu depues de que el usuario este feliz le vas a proporcionar el codigo completo de terraform sin explicaciones, ni comentarios ni nada que no sea el codigo hasta que funcione y luego de que funcione y te agradezca si debes explicar lo que hiciste
    #Notas
    * Recuerda ser lo mas amigable, explicativa y detallada posible
    * Recuerda que cuando ya vayas a generar el codigo solo da el codigo sin comentarios ni explicaciones solo el codigo 
    * Las repuestas de explicaciones o ayudas debe ser en español
    * Si el usuario no te pregunta algo acerca de temas que no tiene relacion con los temas que te mencione, respondele con que la tematica esta fuera del contecto
    * Haz preguntas para saber como puedes ayudarlo de mejor manera y asi no tener inconvenientes"""
     ),
    MessagesPlaceholder(variable_name="messages")
])

memory = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

chain = prompt | llm | StrOutputParser()

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


configuracion = {"configurable": {"session_id": "andres"}}
while True:
    
    pregunta = input("Hola soy Quasar AI escribe tu pregunta (o escribe 'salir' para terminar):\n")
    
    if pregunta.lower() == 'salir':
        print("\n¡Adiós!\n")
        break
    else:
        print(f"\nUsuario: {pregunta}\n")
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=pregunta)], 
             "provider": provider},
            config=configuracion,
        )
        print(f"Quasar AI: {response}\n")
        

                                      


