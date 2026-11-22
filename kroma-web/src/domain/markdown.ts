export type MarkdownInlineNode =
  | { readonly type: 'text'; readonly value: string }
  | { readonly type: 'strong'; readonly children: readonly MarkdownInlineNode[] }
  | { readonly type: 'em'; readonly children: readonly MarkdownInlineNode[] }
  | { readonly type: 'del'; readonly children: readonly MarkdownInlineNode[] }
  | { readonly type: 'code'; readonly value: string }
  | { readonly type: 'link'; readonly href: string; readonly children: readonly MarkdownInlineNode[] };

export type MarkdownBlockNode =
  | {
      readonly type: 'heading';
      readonly level: 1 | 2 | 3 | 4;
      readonly children: readonly MarkdownInlineNode[];
    }
  | { readonly type: 'paragraph'; readonly children: readonly MarkdownInlineNode[] }
  | {
      readonly type: 'list';
      readonly ordered: boolean;
      readonly items: readonly (readonly MarkdownInlineNode[])[];
    }
  | { readonly type: 'blockquote'; readonly children: readonly MarkdownBlockNode[] }
  | { readonly type: 'code_block'; readonly language: string | null; readonly code: string }
  | { readonly type: 'hr' };

export function isSafeMarkdownHref(rawHref: string): boolean {
  const trimmed = rawHref.trim();
  if (!trimmed) return false;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:';
  } catch {
    return false;
  }
}

function isWordChar(ch: string | undefined): boolean {
  return Boolean(ch && /[A-Za-z0-9]/.test(ch));
}

function pushText(nodes: MarkdownInlineNode[], value: string): void {
  if (!value) return;
  const last = nodes[nodes.length - 1];
  if (last && last.type === 'text') {
    nodes[nodes.length - 1] = { type: 'text', value: last.value + value };
  } else {
    nodes.push({ type: 'text', value });
  }
}

function findClosingDelimiter(
  input: string,
  fromIndex: number,
  delimiter: string,
  requireWordBoundaryAfter = false,
): number {
  let idx = fromIndex;
  while (idx <= input.length - delimiter.length) {
    if (input[idx] === '\\') {
      idx += 2;
      continue;
    }
    if (input.startsWith(delimiter, idx)) {
      const prev = input[idx - 1];
      const next = input[idx + delimiter.length];
      if (prev && !/\s/.test(prev) && (!requireWordBoundaryAfter || !isWordChar(next))) {
        return idx;
      }
    }
    idx += 1;
  }
  return -1;
}

export function parseMarkdownInline(input: string): readonly MarkdownInlineNode[] {
  const nodes: MarkdownInlineNode[] = [];
  let i = 0;

  while (i < input.length) {
    const ch = input[i]!;

    // Backslash escape for common punctuation
    if (ch === '\\' && i + 1 < input.length && /[\\`*_{}[\]()#+\-.!~]/.test(input[i + 1]!)) {
      pushText(nodes, input[i + 1]!);
      i += 2;
      continue;
    }

    // Inline code span: `...`
    if (ch === '`') {
      const close = input.indexOf('`', i + 1);
      if (close > i + 1) {
        nodes.push({ type: 'code', value: input.slice(i + 1, close) });
        i = close + 1;
        continue;
      }
    }

    // Markdown link: [label](https://...)
    if (ch === '[') {
      const closeBracket = input.indexOf(']', i + 1);
      if (closeBracket > i + 1 && input[closeBracket + 1] === '(') {
        const closeParen = input.indexOf(')', closeBracket + 2);
        if (closeParen > closeBracket + 2) {
          const label = input.slice(i + 1, closeBracket);
          const href = input.slice(closeBracket + 2, closeParen).trim();
          const children = parseMarkdownInline(label);
          if (isSafeMarkdownHref(href)) {
            nodes.push({ type: 'link', href, children });
          } else {
            nodes.push(...children);
          }
          i = closeParen + 1;
          continue;
        }
      }
    }

    // Bold: **...** or __...__
    if ((input.startsWith('**', i) || input.startsWith('__', i)) && i + 2 < input.length) {
      const delim = input.slice(i, i + 2);
      const isUnderscore = delim === '__';
      const prevChar = i > 0 ? input[i - 1] : undefined;
      const nextChar = input[i + 2];
      if ((!isUnderscore || !isWordChar(prevChar)) && nextChar && !/\s/.test(nextChar)) {
        const close = findClosingDelimiter(input, i + 2, delim, isUnderscore);
        if (close > i + 2) {
          nodes.push({
            type: 'strong',
            children: parseMarkdownInline(input.slice(i + 2, close)),
          });
          i = close + 2;
          continue;
        }
      }
    }

    // Strikethrough: ~~...~~
    if (input.startsWith('~~', i) && i + 2 < input.length && !/\s/.test(input[i + 2]!)) {
      const close = findClosingDelimiter(input, i + 2, '~~');
      if (close > i + 2) {
        nodes.push({
          type: 'del',
          children: parseMarkdownInline(input.slice(i + 2, close)),
        });
        i = close + 2;
        continue;
      }
    }

    // Italic: *...* or _..._ (underscore requires word boundary so w600k_r50.onnx stays intact)
    if ((ch === '*' || ch === '_') && i + 1 < input.length) {
      const isUnderscore = ch === '_';
      const prevChar = i > 0 ? input[i - 1] : undefined;
      const nextChar = input[i + 1];
      if (
        (!isUnderscore || !isWordChar(prevChar)) &&
        nextChar &&
        nextChar !== ch &&
        !/\s/.test(nextChar)
      ) {
        const close = findClosingDelimiter(input, i + 1, ch, isUnderscore);
        if (close > i + 1) {
          nodes.push({
            type: 'em',
            children: parseMarkdownInline(input.slice(i + 1, close)),
          });
          i = close + 1;
          continue;
        }
      }
    }

    pushText(nodes, ch);
    i += 1;
  }

  return nodes;
}

function isHorizontalRule(line: string): boolean {
  return /^\s*(?:[-*_]\s*){3,}$/.test(line);
}

function matchHeading(line: string): { level: 1 | 2 | 3 | 4; text: string } | null {
  const match = /^\s*(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
  if (!match) return null;
  const rawLevel = match[1]!.length;
  const level = (rawLevel <= 4 ? rawLevel : 4) as 1 | 2 | 3 | 4;
  return { level, text: match[2]! };
}

function matchUnorderedItem(line: string): string | null {
  const match = /^\s*[-*+]\s+(.+)$/.exec(line);
  return match ? match[1]! : null;
}

function matchOrderedItem(line: string): string | null {
  const match = /^\s*\d+\.\s+(.+)$/.exec(line);
  return match ? match[1]! : null;
}

function isBlockBoundary(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return true;
  if (trimmed.startsWith('```')) return true;
  if (trimmed.startsWith('>')) return true;
  if (matchHeading(line) !== null) return true;
  if (isHorizontalRule(line)) return true;
  if (matchUnorderedItem(line) !== null) return true;
  if (matchOrderedItem(line) !== null) return true;
  return false;
}

/**
 * Parses GitHub-flavored release note Markdown into a safe, framework-free AST
 * suitable for pure React rendering without `dangerouslySetInnerHTML`.
 */
export function parseMarkdown(source: string): readonly MarkdownBlockNode[] {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const blocks: MarkdownBlockNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i]!;
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // Fenced code block
    const fenceMatch = /^\s*```([A-Za-z0-9_-]*)\s*$/.exec(line);
    if (fenceMatch) {
      const language = fenceMatch[1]?.trim() || null;
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i]!)) {
        codeLines.push(lines[i]!);
        i += 1;
      }
      if (i < lines.length) i += 1; // consume closing ```
      blocks.push({ type: 'code_block', language, code: codeLines.join('\n') });
      continue;
    }

    // Heading
    const heading = matchHeading(line);
    if (heading) {
      blocks.push({
        type: 'heading',
        level: heading.level,
        children: parseMarkdownInline(heading.text),
      });
      i += 1;
      continue;
    }

    // Horizontal rule
    if (isHorizontalRule(line)) {
      blocks.push({ type: 'hr' });
      i += 1;
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i]!.trim().startsWith('>')) {
        quoteLines.push(lines[i]!.replace(/^\s*>\s?/, ''));
        i += 1;
      }
      blocks.push({
        type: 'blockquote',
        children: parseMarkdown(quoteLines.join('\n')),
      });
      continue;
    }

    // Unordered list
    const firstBullet = matchUnorderedItem(line);
    if (firstBullet !== null) {
      const items: (readonly MarkdownInlineNode[])[] = [];
      while (i < lines.length) {
        const bullet = matchUnorderedItem(lines[i]!);
        if (bullet === null) break;
        items.push(parseMarkdownInline(bullet));
        i += 1;
      }
      blocks.push({ type: 'list', ordered: false, items });
      continue;
    }

    // Ordered list
    const firstOrdered = matchOrderedItem(line);
    if (firstOrdered !== null) {
      const items: (readonly MarkdownInlineNode[])[] = [];
      while (i < lines.length) {
        const orderedItem = matchOrderedItem(lines[i]!);
        if (orderedItem === null) break;
        items.push(parseMarkdownInline(orderedItem));
        i += 1;
      }
      blocks.push({ type: 'list', ordered: true, items });
      continue;
    }

    // Paragraph (join consecutive non-boundary lines)
    const paraLines: string[] = [trimmed];
    i += 1;
    while (i < lines.length && !isBlockBoundary(lines[i]!)) {
      paraLines.push(lines[i]!.trim());
      i += 1;
    }
    blocks.push({
      type: 'paragraph',
      children: parseMarkdownInline(paraLines.join(' ')),
    });
  }

  return blocks;
}
