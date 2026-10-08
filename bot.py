import os
import asyncio
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import yt_dlp
from openai import OpenAI

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- الإعدادات ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "your_token")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "your_CHANNEL_ID")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your_api")

# إعداد عميل OpenAI للاتصال بـ OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

DOWNLOAD_DIR = "downloads"
MAX_TELEGRAM_UPLOAD_MB = 49
SEARCH_RESULTS_COUNT = 6
# اعتبار الفيديو "طويل" لو مدته أكتر من 20 دقيقة
LONG_VIDEO_THRESHOLD_SEC = 20 * 60
# مهلة التنزيل: 10 دقائق عشان الفيديوهات الطويلة (ساعة - ساعتين) ياخد وقته
DOWNLOAD_TIMEOUT_SEC = 600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # نقلل السبام في اللوج

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# --- سيرفر بسيط عشان الاستضافة تعرف إن الأبليكيشن شغال (Wispbyte/Render محتاجين Port مفتوح) ---
class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is alive")

    def log_message(self, format, *args):
        pass  # نمنع اللوج بتاع كل ping عشان مايبوظش الكونسول


def run_keep_alive_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), _PingHandler)
    logger.info(f"🌐 Keep-alive server شغال على البورت {port}")
    server.serve_forever()


# --- دالة الـ AI باستخدام نموذج مجاني ---
def get_ai_response(user_query: str):
    """استخدام OpenRouter مع نموذج مجاني للإجابة على أسئلة المستخدم"""
    try:
        completion = client.chat.completions.create(
            model="google/gemma-2-9b-it:free",
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي يساعد المستخدمين في العثور على محتوى فيديو وتقديم نصائح تقنية."},
                {"role": "user", "content": user_query}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"خطأ في الـ AI: {e}")
        return "عذراً، لا أستطيع الإجابة حالياً."


def _format_duration(seconds):
    """تحويل المدة بالثواني لصيغة سهلة القراءة زي 1:23:45"""
    if not seconds:
        return "غير معروفة"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# --- دالة البحث المدمجة (يوتيوب) ---
def _search_youtube_links_sync(query: str, count: int = SEARCH_RESULTS_COUNT):
    """البحث في يوتيوب (synchronous - بتتشغل جوه thread منفصل)
    بيرجع نتائج فيديوهات طويلة وقصيرة مع مدة كل واحد."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
        "socket_timeout": 15,  # عشان مايعلقش لو النت بطيء
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # بنجيب نتايج أكتر عشان نضمن مزيج من فيديوهات طويلة وقصيرة
        results = ydl.extract_info(f"ytsearch{count * 4}:{query}", download=False)
        entries = results.get("entries", []) or []

    query_lower = query.lower().strip()
    matched = [
        v for v in entries
        if v.get("title") and query_lower in v["title"].lower()
    ]
    pool = matched if matched else entries

    short_videos = [v for v in pool if (v.get("duration") or 0) < LONG_VIDEO_THRESHOLD_SEC]
    long_videos = [v for v in pool if (v.get("duration") or 0) >= LONG_VIDEO_THRESHOLD_SEC]

    # بنحاول نديله نص ونص (طويل/قصير) لغاية العدد المطلوب
    half = max(1, count // 2)
    final = short_videos[:half] + long_videos[:count - half]
    if len(final) < count:
        remaining = [v for v in pool if v not in final]
        final += remaining[: count - len(final)]

    return final[:count]


async def search_youtube_links(query: str, count: int = SEARCH_RESULTS_COUNT):
    """نسخة async بتشغل الكود التقيل في thread منفصل عشان مايوقفش البوت كله"""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_search_youtube_links_sync, query, count),
            timeout=25,
        )
    except asyncio.TimeoutError:
        logger.warning("⏱️ البحث في يوتيوب استغرق وقت طويل جداً")
        return []
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        return []


# --- دالة تنزيل الفيديو (synchronous - بتتشغل جوه thread منفصل) ---
def _download_video_sync(url: str):
    """
    بتنزل الفيديو باختيار أفضل جودة ممكنة تفضل تحت حد تيليجرام (~50MB).
    للفيديوهات الطويلة (ساعة/ساعتين) بيتم اختيار جودة أقل تلقائياً عشان تفضل
    تحت الحد المسموح به.
    """
    ydl_opts = {
        "quiet": True,
        # سلسلة فورمات: بيجرب أعلى جودة تحت 48MB، ولو مالقاش يقلل الجودة تدريجياً،
        # ولو برضه مالقاش حاجة تحت الحد يرجع لأقل جودة متاحة (worst) كملاذ أخير.
        "format": (
            "bv*[filesize<48M]+ba[filesize<48M]/"
            "b[filesize<48M]/"
            "b[height<=480][filesize<48M]/"
            "b[height<=360][filesize<48M]/"
            "worst"
        ),
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": 20,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        # لو حصل merge (فيديو+صوت منفصلين) الامتداد بيبقى mp4
        if not os.path.exists(file_path):
            alt_path = os.path.splitext(file_path)[0] + ".mp4"
            if os.path.exists(alt_path):
                file_path = alt_path
    return info, file_path


# --- دالة تنزيل الفيديو ورفعه للقناة ---
async def download_and_send_to_channel(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    status = await update.message.reply_text("⏳ جاري تنزيل الفيديو... (لو الفيديو طويل هياخد وقت أكتر)")
    file_path = None
    try:
        info, file_path = await asyncio.wait_for(
            asyncio.to_thread(_download_video_sync, url),
            timeout=DOWNLOAD_TIMEOUT_SEC,
        )

        if not os.path.exists(file_path):
            await status.edit_text("❌ حصل خطأ أثناء تنزيل الفيديو.")
            return

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > MAX_TELEGRAM_UPLOAD_MB:
            await status.edit_text(
                f"❌ حجم الفيديو كبير جداً ({size_mb:.1f}MB) حتى بعد تقليل الجودة، "
                "ومينفعش يترفع على تيليجرام (الحد الأقصى ~50MB للبوتات)."
            )
            return

        await status.edit_text("📤 جاري الرفع للقناة...")
        title = info.get("title", "فيديو")
        duration_txt = _format_duration(info.get("duration"))
        with open(file_path, "rb") as video_file:
            await context.bot.send_video(
                chat_id=CHANNEL_ID,
                video=video_file,
                caption=f"🎬 {title}\n⏱️ المدة: {duration_txt}",
                supports_streaming=True,
            )
        await status.edit_text("✅ تم تنزيل الفيديو ورفعه للقناة بنجاح.")

    except asyncio.TimeoutError:
        await status.edit_text("❌ التنزيل استغرق وقت طويل جداً وتم إلغاؤه (الفيديو كبير أو الرابط بطيء).")
    except Exception as e:
        logger.error(f"خطأ أثناء التنزيل/الرفع: {e}")
        await status.edit_text("❌ حصل خطأ أثناء تنزيل أو رفع الفيديو.")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


# --- أمر /download ---
async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["awaiting_download_link"] = True
    await update.message.reply_text("🔗 ابعتلي رابط الفيديو اللي عايز تنزله وترفعه للقناة.")


# --- معالجة الرسائل ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # لو البوت مستني رابط بعد /download
    if context.user_data.get("awaiting_download_link"):
        context.user_data["awaiting_download_link"] = False
        if text.startswith("http://") or text.startswith("https://"):
            await download_and_send_to_channel(update, context, text)
        else:
            await update.message.reply_text("⚠️ ده مش رابط صحيح. ابعت /download تاني وحط الرابط.")
        return

    # لو الرسالة رابط مباشرة (من غير /download) نتعامل معاها كتنزيل برضه
    if text.startswith("http://") or text.startswith("https://"):
        await download_and_send_to_channel(update, context, text)
        return

    # غير كده: أي كلمة عادية تعتبر بحث في يوتيوب
    status = await update.message.reply_text("🎉 يوهو! بدور على فيديوهات ليك...")
    results = await search_youtube_links(text)

    if results:
        lines = ["🎬 *نتائج البحث (فيديوهات طويلة وقصيرة):*"]
        for v in results:
            video_url = v.get("url") or f"https://www.youtube.com/watch?v={v.get('id')}"
            title = v.get("title", "بدون عنوان")
            duration = v.get("duration") or 0
            duration_txt = _format_duration(duration)
            tag = "🕒 طويل" if duration >= LONG_VIDEO_THRESHOLD_SEC else "⚡ قصير"
            lines.append(f"- [{title}]({video_url})\n   {tag} | المدة: {duration_txt}")
        reply = "\n".join(lines)
        await status.edit_text(reply, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
    else:
        await status.edit_text("لم أجد نتائج، هحاول أرد عليك بالـ AI بدل كده...")
        try:
            ai_reply = await asyncio.wait_for(
                asyncio.to_thread(get_ai_response, text), timeout=30
            )
        except asyncio.TimeoutError:
            ai_reply = "عذراً، استغرق الرد وقت طويل جداً."
        await update.message.reply_text(ai_reply)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "مرحباً! 👋\n"
        "- ابعتلي أي كلمة وهبحثلك عنها في يوتيوب وأديك فيديوهات طويلة وقصيرة.\n"
        "- ابعت /download وبعدها رابط الفيديو، هنزله وأرفعه في القناة (حتى لو مدته ساعة أو ساعتين، "
        "هيتم اختيار جودة مناسبة تفضل تحت حد تيليجرام)."
    )


async def run_bot():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("download", download_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    logger.info("✅ البوت شغال...")
    # نسيب البوت شغال لحد ما حد يوقفه (Ctrl+C أو الاستضافة توقف الكونتينر)
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main():
    # نشغل السيرفر الوهمي في thread منفصل عشان الاستضافة تعتبر الأبليكيشن شغال
    threading.Thread(target=run_keep_alive_server, daemon=True).start()

    # asyncio.run() بينشئ ويدير الـ event loop بنفسه من غير ما يستخدم
    # asyncio.get_event_loop() المكسورة في بايثون 3.14
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
