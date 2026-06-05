import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const indexPath = path.join(__dirname, 'dist', 'index.html');
let html = fs.readFileSync(indexPath, 'utf8');

html = html.replace(/<script type="module" crossorigin>/g, '<script>');
html = html.replace(/<script type="module" crossorigin src=/g, '<script src=');
html = html.replace(/<script type="module">/g, '<script>');
html = html.replace(/ crossorigin/g, '');

const scriptMatch = html.match(/(<script>[\s\S]*?<\/script>)/);
if (scriptMatch) {
  html = html.replace(scriptMatch[0], '');
  html = html.replace('</body>', () => `${scriptMatch[0]}\n  </body>`);
}

fs.writeFileSync(indexPath, html);
console.log('Post-build: Processed index.html for Sequencer');
