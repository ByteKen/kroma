import { Fragment, useMemo, type ReactNode } from 'react';
import {
  parseMarkdown,
  type MarkdownBlockNode,
  type MarkdownInlineNode,
} from '../../domain/markdown';
import { openExternalUrl } from '../../domain/platform';

interface MarkdownViewProps {
  readonly markdown: string;
  readonly className?: string;
}

function renderInlineNodes(nodes: readonly MarkdownInlineNode[], keyPrefix: string): ReactNode {
  return nodes.map((node, index) => {
    const key = `${keyPrefix}-i${index}`;
    switch (node.type) {
      case 'text':
        return <Fragment key={key}>{node.value}</Fragment>;
      case 'strong':
        return (
          <strong key={key} className="font-semibold text-ink-primary">
            {renderInlineNodes(node.children, key)}
          </strong>
        );
      case 'em':
        return (
          <em key={key} className="italic text-ink-primary/90">
            {renderInlineNodes(node.children, key)}
          </em>
        );
      case 'del':
        return (
          <del key={key} className="line-through text-ink-telemetry">
            {renderInlineNodes(node.children, key)}
          </del>
        );
      case 'code':
        return (
          <code
            key={key}
            className="tabular-nums-mono text-[11px] px-1.5 py-0.5 rounded bg-canvas border border-hairline text-amber-soft"
          >
            {node.value}
          </code>
        );
      case 'link':
        return (
          <a
            key={key}
            href={node.href}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => {
              e.preventDefault();
              void openExternalUrl(node.href);
            }}
            className="text-amber-soft underline decoration-amber-core/50 underline-offset-2 hover:text-amber-core transition-colors cursor-pointer"
          >
            {renderInlineNodes(node.children, key)}
          </a>
        );
    }
  });
}

function renderBlockNodes(blocks: readonly MarkdownBlockNode[], keyPrefix: string): ReactNode {
  return blocks.map((block, index) => {
    const key = `${keyPrefix}-b${index}`;
    switch (block.type) {
      case 'heading': {
        const headingClass =
          block.level === 1
            ? 'font-display text-[16px] font-semibold text-ink-primary tracking-tight mt-4 first:mt-0'
            : block.level === 2
              ? 'font-display text-[14px] font-semibold text-ink-primary tracking-tight mt-4 first:mt-0 pb-1 border-b border-hairline/70'
              : block.level === 3
                ? 'font-display text-[12.5px] font-semibold text-ink-primary mt-3 first:mt-0'
                : 'tabular-nums-mono text-[11px] uppercase tracking-wider text-ink-telemetry font-semibold mt-3 first:mt-0';
        const Tag = (`h${block.level}`) as 'h1' | 'h2' | 'h3' | 'h4';
        return (
          <Tag key={key} className={headingClass}>
            {renderInlineNodes(block.children, key)}
          </Tag>
        );
      }
      case 'paragraph':
        return (
          <p key={key} className="text-[12.5px] leading-relaxed text-ink-secondary">
            {renderInlineNodes(block.children, key)}
          </p>
        );
      case 'list': {
        const ListTag = block.ordered ? 'ol' : 'ul';
        return (
          <ListTag
            key={key}
            className={`pl-5 space-y-1.5 text-[12.5px] leading-relaxed text-ink-secondary ${
              block.ordered ? 'list-decimal' : 'list-disc'
            } marker:text-amber-core/70`}
          >
            {block.items.map((item, itemIdx) => (
              <li key={`${key}-li${itemIdx}`}>
                {renderInlineNodes(item, `${key}-li${itemIdx}`)}
              </li>
            ))}
          </ListTag>
        );
      }
      case 'blockquote':
        return (
          <blockquote
            key={key}
            className="pl-3.5 py-1 border-l-2 border-amber-core/60 bg-amber-core/[0.04] rounded-r-md space-y-1.5 text-ink-secondary"
          >
            {renderBlockNodes(block.children, key)}
          </blockquote>
        );
      case 'code_block':
        return (
          <pre
            key={key}
            className="p-3 rounded-lg bg-canvas border border-hairline overflow-x-auto tabular-nums-mono text-[11px] leading-relaxed text-ink-primary"
          >
            <code>{block.code}</code>
          </pre>
        );
      case 'hr':
        return <hr key={key} className="border-0 border-t border-hairline my-3" />;
    }
  });
}

export function MarkdownView({ markdown, className = '' }: MarkdownViewProps) {
  const blocks = useMemo(() => parseMarkdown(markdown), [markdown]);

  return (
    <div className={`space-y-2.5 ${className}`}>
      {renderBlockNodes(blocks, 'md')}
    </div>
  );
}
