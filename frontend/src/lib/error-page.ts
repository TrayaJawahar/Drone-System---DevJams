export function renderErrorPage(): string {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><title>This page didn't load</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>body{font:15px/1.5 system-ui;background:#fafafa;color:#111;display:grid;place-items:center;min-height:100vh;margin:0;padding:1.5rem}.card{max-width:28rem;text-align:center;padding:2rem}p{color:#4b5563}.actions{display:flex;gap:.5rem;justify-content:center}a,button{padding:.5rem 1rem;border-radius:.375rem;font:inherit;cursor:pointer;text-decoration:none}.primary{background:#111;color:#fff}.secondary{background:#fff;color:#111;border:1px solid #d1d5db}</style>
</head><body><div class="card"><h1>This page didn't load</h1><p>Something went wrong on our end. You can try refreshing or head back home.</p><div class="actions"><button class="primary" onclick="location.reload()">Try again</button><a class="secondary" href="/">Go home</a></div></div></body></html>`;
}
