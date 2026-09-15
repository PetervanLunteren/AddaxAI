/**
 * AddaxAI model download relay.
 *
 * A Cloudflare Worker that forwards requests for the Addax-Data-Science
 * HuggingFace repos to huggingface.co and streams the answer back. It
 * stores nothing, so a new model or a fixed inference.py on HuggingFace is
 * live here the same second. It exists for users whose network blocks
 * huggingface.co and *.hf.co outright (a US state web filter categorised
 * them as "AI content", 2026-09-11): the app downloads from HuggingFace
 * first and retries through this relay only when the network answered
 * with a block page (NetworkBlockedError in backend/app/ml/hf_downloader.py).
 *
 * Deploy: Cloudflare dashboard, Workers & Pages, create a Worker named
 * addaxai-models, paste this file, Deploy. Free plan, no card. The URL is
 * https://addaxai-models.<account>.workers.dev and is the default
 * `hf_fallback_url` in backend/app/core/config.py.
 *
 * What it answers, all under https://huggingface.co:
 *   GET/HEAD /Addax-Data-Science/<id>/resolve/<rev>/<path>   the files
 *   GET/POST /api/models/Addax-Data-Science/<id>[/...]       tree, paths-info, info
 *   GET      /                                               a one-line health text
 * Everything else is refused with 404, so this is a relay for our own
 * repos and not a general HuggingFace proxy.
 */

const UPSTREAM = "https://huggingface.co";
const ORG = "Addax-Data-Science";
const ALLOWED_PREFIXES = [`/${ORG}/`, `/api/models/${ORG}/`];

// Headers that must not be forwarded: hop-by-hop and Cloudflare-added ones,
// and the client's credentials. Our repos are public, so no request here
// ever needs a token, and a user with HF_TOKEN in their environment must
// not have it travel through this relay.
const STRIP_REQUEST_HEADERS = ["host", "cookie", "authorization", "cf-connecting-ip", "cf-ray", "cf-visitor", "cf-ipcountry", "x-forwarded-for", "x-forwarded-proto", "x-real-ip"];

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response(
        "AddaxAI model download relay is working. " +
          "It forwards model downloads to huggingface.co for the AddaxAI app.\n",
        { headers: { "content-type": "text/plain; charset=utf-8" } },
      );
    }

    if (!ALLOWED_PREFIXES.some((p) => url.pathname.startsWith(p))) {
      return new Response("Not found\n", { status: 404, headers: { "content-type": "text/plain" } });
    }
    if (!["GET", "HEAD", "POST"].includes(request.method)) {
      return new Response("Method not allowed\n", { status: 405, headers: { "content-type": "text/plain" } });
    }

    const headers = new Headers(request.headers);
    for (const name of STRIP_REQUEST_HEADERS) headers.delete(name);

    // Follow the redirect to the *.hf.co CDN here, so the client only ever
    // talks to this host. That is the whole point: the client's network
    // blocks the CDN too.
    const upstream = await fetch(UPSTREAM + url.pathname + url.search, {
      method: request.method,
      headers,
      body: request.method === "POST" ? request.body : undefined,
      redirect: "follow",
    });

    // Stream the body through untouched. Status and headers (including
    // Content-Length, Content-Range, ETag and Accept-Ranges, which the
    // app's parallel range downloader relies on) are copied as they came.
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: upstream.headers,
    });
  },
};
