import httpx
from twitchio.ext import commands

API_URL = "http://127.0.0.1:8000/predict"

BOT_TOKEN = "oauth:jj1syyde8jxv1eltyapltsl5y9p9yh"
CLIENT_ID = "8r4ltdorvf2pl6i73ewhfpx9kueznp"
BROADCASTER_ID = "1452332459"
MODERATOR_ID = "1452332459"
CHANNEL = "ceqw3bot"


class Bot(commands.Bot):

    def __init__(self):
        super().__init__(
            token=BOT_TOKEN,
            prefix="!",
            initial_channels=[CHANNEL]
        )

    async def event_ready(self):
        print(f"Bot conectado como {self.nick}")

    async def delete_message(self, message):
        """Borra un mensaje usando la API REST de Twitch (Helix)"""
        try:
            token = BOT_TOKEN.replace("oauth:", "")

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    "https://api.twitch.tv/helix/moderation/chat",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Client-Id": CLIENT_ID,
                    },
                    params={
                        "broadcaster_id": BROADCASTER_ID,
                        "moderator_id": MODERATOR_ID,
                        "message_id": message.id,
                    }
                )

            if response.status_code == 204:
                print(f"Mensaje de {message.author.name} borrado correctamente.")
            else:
                print(f"Error al borrar: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error al borrar mensaje: {e}")

    async def event_message(self, message):
        if message.echo:
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    API_URL,
                    json={"message": message.content},
                    timeout=5.0
                )

            result = response.json()

            label = result["label"]
            confidence = result["confidence"]
            probs = result.get("probabilities", {})

            hate_prob  = probs.get("hate",  confidence if label == "hate"  else 0.0)
            toxic_prob = probs.get("toxic", confidence if label == "toxic" else 0.0)
            combined   = hate_prob + toxic_prob

            print(f"{message.author.name}: {message.content}")
            print(f"Clasificacion: {label} ({confidence:.2f}) | hate={hate_prob:.2f} toxic={toxic_prob:.2f} combinado={combined:.2f}")

            # Borrar si la suma de hate + toxic supera 80%
            if combined > 0.80:
                await self.delete_message(message)
                await message.channel.send(
                    f"@{message.author.name} tu mensaje fue eliminado por lenguaje inapropiado."
                )

            # Advertencia si la suma supera 50% pero no llega al 80%
            elif combined > 0.50:
                await message.channel.send(
                    f"@{message.author.name} cuidado con el lenguaje."
                )

        except Exception as e:
            print("Error llamando a la API:", e)

        await self.handle_commands(message)

    @commands.command()
    async def ping(self, ctx):
        await ctx.send("hola")


bot = Bot()
bot.run()