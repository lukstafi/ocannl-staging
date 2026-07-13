-- Move the "Abstract" section of the workshop article into pandoc
-- metadata, so the LaTeX template renders it as \begin{abstract}
-- instead of a section.
-- Note: lua filters see the document BEFORE --shift-heading-level-by
-- is applied, so the header levels here are the Markdown source levels
-- (## Abstract is level 2). Matching is level-agnostic: the section
-- ends at the next header of the same or shallower level.
local abstract = {}
local abs_level = nil

local function harvest(blocks)
  local out = {}
  for _, b in ipairs(blocks) do
    if b.t == 'Header' then
      if abs_level and b.level <= abs_level then
        abs_level = nil
      end
      if pandoc.utils.stringify(b.content) == 'Abstract' then
        abs_level = b.level
      else
        table.insert(out, b)
      end
    elseif abs_level then
      table.insert(abstract, b)
    else
      table.insert(out, b)
    end
  end
  return out
end

function Pandoc(doc)
  doc.blocks = harvest(doc.blocks)
  if #abstract > 0 then
    doc.meta.abstract = pandoc.MetaBlocks(abstract)
  end
  return doc
end
